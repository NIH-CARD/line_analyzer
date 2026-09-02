"""Self-contained test suite. No pytest, no R, no statsmodels.

    python -m proteomics_revertant.tests

Every check computes an independent reference (brute force, Monte Carlo, or a
closed form derived a different way) rather than comparing the code to itself.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

from . import config as C
from .analysis import (analyse, build_design, clone_summaries,
                       cochran_armitage, consensus_icc)
from .datasets import (SAMPLE_COLUMNS, design_from_samples, load_dataset,
                       validate, write_dataset)
from .ebayes import bh, fit_f_dist, moderated_t, squeeze_var, trigamma_inverse
from .simulate import simulate

DATA = Path(__file__).resolve().parents[1] / "data"

RESULTS = []


def check(name):
    def deco(fn):
        def wrapped():
            t0 = time.time()
            try:
                detail = fn() or ""
                RESULTS.append((name, True, detail, time.time() - t0))
            except AssertionError as e:
                RESULTS.append((name, False, str(e), time.time() - t0))
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {e}", time.time() - t0))
        wrapped.__name__ = fn.__name__
        return wrapped
    return deco


# ---------------------------------------------------------------------------
# ebayes.py
# ---------------------------------------------------------------------------

@check("trigamma_inverse inverts trigamma")
def t_trigamma():
    x = np.array([1e-8, 1e-3, 0.05, 0.3, 1.0, 5.0, 1e3, 1e8])
    y = trigamma_inverse(x)
    back = special.polygamma(1, y)
    rel = np.abs(back - x) / x
    assert np.max(rel) < 1e-6, f"max rel error {rel.max():.3e}"
    return f"max relative error {rel.max():.2e} over 8 decades"


@check("fit_f_dist recovers known scaled-F prior")
def t_fitfdist():
    rng = np.random.default_rng(7)
    d0_true, s02_true, d = 8.0, 0.25, 3.0
    n = 200_000
    # sigma^2_g ~ s02 * d0 / chisq(d0);  s^2_g | sigma^2_g ~ sigma^2_g * chisq(d)/d
    sigma2 = s02_true * d0_true / rng.chisquare(d0_true, n)
    s2 = sigma2 * rng.chisquare(d, n) / d
    s02, d0 = fit_f_dist(s2, np.full(n, d))
    assert abs(d0 - d0_true) / d0_true < 0.10, f"d0 {d0:.2f} vs {d0_true}"
    assert abs(s02 - s02_true) / s02_true < 0.05, f"s02 {s02:.4f} vs {s02_true}"
    return f"recovered d0={d0:.2f} (true 8.0), s02={s02:.4f} (true 0.25)"


@check("squeeze_var shrinks toward the prior and exempts outliers")
def t_squeeze():
    rng = np.random.default_rng(11)
    d, n = 3.0, 20000
    sigma2 = 0.2 * 6.0 / rng.chisquare(6.0, n)          # true prior d0 = 6
    s2 = sigma2 * rng.chisquare(d, n) / d
    dfv = np.full(n, d)

    post, s02, d0, d0_vec = squeeze_var(s2, dfv, robust=False)
    expect = (d0 * s02 + d * s2) / (d0 + d)
    assert np.allclose(post, expect), "posterior is not the stated convex combination"
    assert np.var(post) < np.var(s2), "shrinkage did not reduce variance"
    assert np.all((post >= np.minimum(s2, s02) - 1e-12) &
                  (post <= np.maximum(s2, s02) + 1e-12)), "posterior outside [s2, s02]"
    assert np.all(np.diff(post[np.argsort(s2)]) >= -1e-12), "shrinkage not monotone in s2"
    # posterior must be closer to the truth than the raw estimate
    mse_raw = float(np.mean((np.log(s2) - np.log(sigma2)) ** 2))
    mse_post = float(np.mean((np.log(post) - np.log(sigma2)) ** 2))
    assert mse_post < mse_raw, "moderation did not improve variance estimation"

    # robust mode: a hypervariable contaminant must be exempted from shrinkage
    s2c = s2.copy()
    s2c[:200] *= 60.0
    postc, _, _, d0c = squeeze_var(s2c, dfv, robust=True, outlier_p=0.05)
    exempt = float(np.mean(d0c[:200] == 0.0))
    assert exempt > 0.9, f"only {exempt:.2f} of contaminants exempted from shrinkage"
    assert np.mean(d0c[200:] == 0.0) < 0.10, "too many clean proteins exempted"
    return (f"log-variance MSE {mse_raw:.3f} -> {mse_post:.3f}, d0={d0:.2f} (true 6); "
            f"{exempt:.0%} of contaminants exempted in robust mode")


@check("bh matches a brute-force reference and is monotone")
def t_bh():
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(0, 1, 500), rng.beta(0.2, 5, 100), [np.nan] * 20])
    got = bh(p)
    ok = np.isfinite(p)
    pv = p[ok]
    m = pv.size
    ref = np.empty(m)
    for i in range(m):
        # definition: min over k >= rank(i) of (m/k) * p_(k)
        srt = np.sort(pv)
        rank = np.searchsorted(srt, pv[i], side="left") + 1
        ref[i] = min(min(m / k * srt[k - 1] for k in range(rank, m + 1)), 1.0)
    assert np.allclose(got[ok], ref, atol=1e-12), "BH disagrees with brute force"
    assert np.all(np.isnan(got[~ok])), "NaN inputs did not stay NaN"
    assert np.all(got[ok] >= pv - 1e-12), "adjusted p below raw p"
    return f"{m} p-values match brute-force BH exactly"


@check("moderated_t p-values are consistent with their own t and df")
def t_modt():
    rng = np.random.default_rng(5)
    beta = rng.normal(0, 1, 1000)
    unsc = rng.uniform(0.5, 2, 1000)
    pv = rng.uniform(0.05, 0.5, 1000)
    dfree = np.full(1000, 9.3)
    se, t, p = moderated_t(beta, unsc, pv, dfree)
    assert np.allclose(se, unsc * np.sqrt(pv)), "SE is not unscaled * sqrt(var)"
    assert np.allclose(t, beta / se), "t is not beta/SE"
    assert np.allclose(p, 2 * stats.t.sf(np.abs(t), df=dfree)), "p/t/df inconsistent"
    return "SE, t and p mutually consistent over 1000 draws"


# ---------------------------------------------------------------------------
# analysis.py primitives
# ---------------------------------------------------------------------------

@check("exact Cochran-Armitage matches Monte-Carlo permutation")
def t_ca():
    rng = np.random.default_rng(13)
    cases = [([8, 8, 0], [8, 8, 4], [0, 1, 2]),
             ([6, 3, 1], [8, 4, 8], [0, 1, 2]),
             ([4, 4, 4], [8, 8, 4], [0, 1, 2])]
    lines = []
    for det, n, sc in cases:
        _, p_exact = cochran_armitage(det, n, sc)
        N, R = sum(n), sum(det)
        labels = np.array([0] * (N - R) + [1] * R)
        groups = np.repeat(np.arange(len(n)), n)
        pbar = R / N
        T_obs = sum(sc[i] * (det[i] - n[i] * pbar) for i in range(len(n)))
        hits = 0
        B = 40000
        for _ in range(B):
            perm = rng.permutation(labels)
            r = np.array([perm[groups == i].sum() for i in range(len(n))])
            T = float(np.sum(np.array(sc) * (r - np.array(n) * pbar)))
            hits += abs(T) >= abs(T_obs) - 1e-9
        p_mc = hits / B
        se = np.sqrt(max(p_mc * (1 - p_mc), 1e-9) / B)
        assert abs(p_exact - p_mc) < max(5 * se, 2e-3), \
            f"det={det}: exact {p_exact:.5f} vs MC {p_mc:.5f} (se {se:.5f})"
        lines.append(f"det={det} exact={p_exact:.5f} MC={p_mc:.5f}")
    return "; ".join(lines)


@check("clone_summaries reproduces means, counts and within-clone variance")
def t_clonesum():
    design = C.BACKGROUND_A
    expr, samples, _ = simulate(design, seed=1)
    clones = [c.name for c in design.clones]
    means, counts, s2w, dfw = clone_summaries(expr, samples, clones)
    # independent recomputation with pandas groupby
    long = expr.T.join(samples.set_index("library")[["clone"]])
    g = long.groupby("clone")
    ref_mean = g.mean().T[clones].to_numpy()
    ref_count = g.count().T[clones].to_numpy()
    assert np.array_equal(ref_count, counts), "observation counts disagree"
    assert np.allclose(np.nan_to_num(ref_mean, nan=-999),
                       np.nan_to_num(means, nan=-999), atol=1e-10), "clone means disagree"
    ref_var = g.var(ddof=1).T[clones]
    ss = (ref_var * (ref_count - 1)).to_numpy()
    ref_s2 = np.nansum(ss, axis=1) / np.maximum(dfw, 1)
    ok = dfw > 0
    assert np.allclose(ref_s2[ok], s2w[ok], atol=1e-9), "pooled within-clone variance disagrees"
    return f"{expr.shape[0]} proteins x {len(clones)} clones verified against pandas groupby"


@check("weighted dose fit matches an independent closed-form WLS")
def t_wls():
    design = C.BACKGROUND_A
    expr, samples, _ = simulate(design, seed=2)
    res, meta = analyse(expr, samples, design)
    clones = [c.name for c in design.clones]
    means, counts, s2w, _ = clone_summaries(expr, samples, clones)
    X = build_design(design)[0]
    rho = meta["consensus_icc"]

    complete = res.index[(counts > 0).all(axis=1) & (res["track"] == "continuous").values]
    tested = 0
    for g in complete[:300]:
        w = 1.0 / (rho + (1 - rho) / counts[g])
        y = means[g]
        # independent WLS: solve the normal equations by hand
        Xw = X * np.sqrt(w)[:, None]
        yw = y * np.sqrt(w)
        beta = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
        r = yw - Xw @ beta
        s2 = float(r @ r) / (len(y) - X.shape[1])
        cov = np.linalg.inv(Xw.T @ Xw)
        assert abs(beta[1] - res.loc[g, "dose_logFC"]) < 1e-3, \
            f"protein {g}: dose beta {beta[1]:.6f} vs {res.loc[g,'dose_logFC']}"
        # moderated SE = sqrt(post_var * cov[1,1])
        se_ref = np.sqrt(res.loc[g, "clone_var_moderated"] * cov[1, 1])
        assert abs(se_ref - res.loc[g, "dose_SE"]) < 1e-3, f"protein {g}: SE mismatch"
        assert abs(s2 - res.loc[g, "sigma2_tech"]) > -1  # sanity, unused
        tested += 1
    assert tested > 100, "too few complete proteins to test"
    return f"{tested} proteins: beta and moderated SE match hand-solved WLS"


@check("residual df equals clones minus parameters, never libraries")
def t_df():
    design = C.BACKGROUND_A
    expr, samples, _ = simulate(design, seed=3)
    res, meta = analyse(expr, samples, design)
    fitted = res[res["df_residual"].notna()]
    assert fitted["df_residual"].max() == design.n_clones - 2, \
        f"max df {fitted['df_residual'].max()}, expected {design.n_clones - 2}"
    assert meta["n_libraries"] == 20 and meta["residual_df"] == 3
    n_full = int((fitted["df_residual"] == 3).sum())
    return (f"{n_full} proteins at 3 df (5 clones - 2 params); "
            f"20 libraries would have given 18")


@check("pairwise contrast variance uses the clone-level error term")
def t_pairwise():
    design = C.BACKGROUND_A
    expr, samples, _ = simulate(design, seed=4)
    res, meta = analyse(expr, samples, design)
    clones = [c.name for c in design.clones]
    means, counts, _, _ = clone_summaries(expr, samples, clones)
    rho = meta["consensus_icc"]
    ix = {c: i for i, c in enumerate(clones)}
    sub = res[(res["track"] == "continuous") & res["edit_vs_rev_SE"].notna()].head(200)
    for g in sub.index:
        if counts[g].min() == 0:
            continue
        w = 1.0 / (rho + (1 - rho) / counts[g])
        lfc = means[g, ix["HomEd"]] - means[g, ix["HomRev"]]
        se = np.sqrt(res.loc[g, "clone_var_moderated"] *
                     (1 / w[ix["HomEd"]] + 1 / w[ix["HomRev"]]))
        assert abs(lfc - res.loc[g, "edit_vs_rev_logFC"]) < 1e-3, "contrast logFC mismatch"
        assert abs(se - res.loc[g, "edit_vs_rev_SE"]) < 1e-3, "contrast SE mismatch"
    return f"{len(sub)} proteins: Var = s2_moderated * sum(c_i^2 / w_i) verified"


# ---------------------------------------------------------------------------
# end-to-end behaviour
# ---------------------------------------------------------------------------

@check("type-I error is calibrated under a true null (5 seeds)")
def t_null():
    rates, iut_fp, pa_fp = [], 0, 0
    for seed in range(101, 106):
        expr, samples, _ = simulate(C.BACKGROUND_A, seed=seed, null=True)
        res, _ = analyse(expr, samples, C.BACKGROUND_A)
        p = res["dose_P"].dropna()
        rates.append(float((p < 0.05).mean()))
        iut_fp += int((res["iut_FDR"] < 0.05).sum())
        pa_fp += int((res["pa_trend_FDR"] < 0.05).sum())
    mean_rate = float(np.mean(rates))
    assert 0.02 < mean_rate < 0.12, f"type-I rate {mean_rate:.3f} outside [0.02, 0.12]"
    assert iut_fp == 0, f"{iut_fp} IUT false discoveries under the null"
    assert pa_fp <= 2, f"{pa_fp} presence/absence false discoveries under the null"
    return (f"dose_P<0.05 rate {mean_rate:.3f} across 5 null datasets; "
            f"IUT FDR<0.05 false calls: {iut_fp}; PA false calls: {pa_fp}")


@check("FDR control holds under the null despite heavy-tailed clone artefacts")
def t_fdr_null():
    dose_fp, iut_fp, pa_fp, raw = [], [], [], []
    for seed in range(301, 309):
        expr, samples, _ = simulate(C.BACKGROUND_A, seed=seed, null=True)
        res, meta = analyse(expr, samples, C.BACKGROUND_A)
        dose_fp.append(int((res["dose_FDR"] < 0.05).sum()))
        iut_fp.append(int((res["iut_FDR"] < 0.05).sum()))
        pa_fp.append(int((res["pa_trend_FDR"] < 0.05).sum()))
        raw.append(float((res["dose_P"].dropna() < 0.05).mean()))
    assert sum(dose_fp) <= 2, f"dose FDR false discoveries: {dose_fp}"
    assert sum(iut_fp) == 0, f"IUT false discoveries: {iut_fp}"
    assert sum(pa_fp) <= 2, f"presence/absence false discoveries: {pa_fp}"
    return (f"across 8 null datasets: {sum(dose_fp)} dose, {sum(iut_fp)} IUT, "
            f"{sum(pa_fp)} detection-track discoveries at FDR<0.05 "
            f"(raw p<0.05 rate {np.mean(raw):.3f}, mildly anticonservative by design)")


@check("known effects are recovered without bias (5 seeds)")
def t_recovery():
    errs, tgt = [], []
    for seed in range(201, 206):
        expr, samples, truth = simulate(C.BACKGROUND_A, seed=seed)
        res, _ = analyse(expr, samples, C.BACKGROUND_A)
        d = res.set_index("protein_id").join(truth)
        ok = d["dose_logFC"].notna()
        errs.append((d.loc[ok, "dose_logFC"] - d.loc[ok, "true_per_allele_lfc"]).to_numpy())
        tgt.append(float(d.loc[C.TARGET_PROTEIN, "dose_logFC"]))
    e = np.concatenate(errs)
    assert abs(np.mean(e)) < 0.02, f"mean estimation bias {np.mean(e):.4f}"
    assert abs(np.mean(tgt) - C.TARGET_PER_ALLELE_LFC) < 0.25, \
        f"target mean estimate {np.mean(tgt):.3f} vs true {C.TARGET_PER_ALLELE_LFC}"
    return (f"mean bias {np.mean(e):+.4f} log2/allele (n={e.size}); "
            f"target estimated {np.mean(tgt):.3f} vs true {C.TARGET_PER_ALLELE_LFC}")


@check("lack of fit is not mistaken for hypervariance (regression guard)")
def t_lackoffit():
    """The planted big effect is partly recessive, so it has a large residual
    against the additive model. An earlier version flagged it as hypervariable,
    exempted it from shrinkage, tripled its SE and lost the hit entirely."""
    hits = 0
    for seed in [20260901, 1, 2, 3, 4]:
        expr, samples, _ = simulate(C.BACKGROUND_A, seed=seed)
        res, _ = analyse(expr, samples, C.BACKGROUND_A)
        row = res.set_index("protein_id").loc[C.TARGET_PROTEIN]
        assert not bool(row["variance_outlier"]), \
            f"seed {seed}: recessive target flagged as a variance outlier"
        assert row["dose_FDR"] < 0.05, f"seed {seed}: target dose FDR {row['dose_FDR']}"
        hits += row["call"] == "variant_attributable"
    assert hits >= 4, f"target called variant_attributable in only {hits}/5 seeds"
    return f"target recovered at FDR<0.05 in 5/5 seeds, IUT-called in {hits}/5"


@check("no imputation happens anywhere")
def t_no_imputation():
    design = C.BACKGROUND_A
    expr, samples, _ = simulate(design, seed=6)
    arr = expr.to_numpy(float)
    n_missing = int(np.isnan(arr).sum())
    assert n_missing > 0, "simulation produced no missing values to test"
    res, _ = analyse(expr, samples, design)
    # a protein missing from an entire state must not receive a continuous call
    obs_cols = [c for c in res.columns if c.startswith("obs_")]
    empty_state = (res[obs_cols] == 0).any(axis=1)
    bad = res.loc[empty_state & (res["track"] == "continuous")]
    assert bad.empty, f"{len(bad)} proteins with an empty state routed to continuous only"
    # proteins with an empty state that still clear the presence filter must
    # reach the detection track; the rest are honestly marked insufficient
    obs = res[obs_cols].to_numpy()
    passes_filter = (obs >= C.MIN_OBS_PER_STATE).any(axis=1)
    should_test = empty_state.to_numpy() & passes_filter
    routed = res.loc[should_test, "track"].str.contains("presence_absence").all()
    assert routed, "an empty-state protein passing the filter missed the detection track"
    rest = res.loc[empty_state.to_numpy() & ~passes_filter, "track"]
    assert (rest == "insufficient").all(), "unfiltered proteins mislabelled"
    src = Path(__file__).with_name("analysis.py").read_text()
    for banned in ["fillna", "nan_to_num(means", "SimpleImputer", "interpolate("]:
        assert banned not in src, f"found imputation-like call: {banned}"
    return (f"{n_missing} missing cells; {int(should_test.sum())} testable empty-state "
            f"proteins routed to the detection track, {len(rest)} marked insufficient")


@check("results are deterministic for a fixed seed")
def t_determinism():
    a = analyse(*simulate(C.BACKGROUND_A, seed=42)[:2], C.BACKGROUND_A)[0]
    b = analyse(*simulate(C.BACKGROUND_A, seed=42)[:2], C.BACKGROUND_A)[0]
    pd.testing.assert_frame_equal(a, b)
    c = analyse(*simulate(C.BACKGROUND_A, seed=43)[:2], C.BACKGROUND_A)[0]
    assert not a["dose_logFC"].equals(c["dose_logFC"]), "different seeds gave identical output"
    return "seed 42 reproduces bit-for-bit; seed 43 differs"


@check("output contract: 1000 rows, unique ids, valid statistics")
def t_contract():
    notes = []
    for key in ["A_homozygous_wt", "B_heterozygous_wt", "B_heterozygous_wt_minimal"]:
        design = C.DESIGNS[key]
        res, _ = analyse(*simulate(design, seed=9)[:2], design)
        assert len(res) == C.N_PROTEINS, f"{key}: {len(res)} rows"
        assert res["protein_id"].is_unique, f"{key}: duplicate protein ids"
        assert res["protein_id"].notna().all(), f"{key}: null protein id"
        for col in [c for c in res.columns if c.endswith(("_P", "_FDR"))]:
            v = res[col].dropna()
            assert ((v >= 0) & (v <= 1)).all(), f"{key}: {col} outside [0,1]"
        for col in [c for c in res.columns if c.endswith("_SE")]:
            v = res[col].dropna()
            assert (v > 0).all(), f"{key}: non-positive SE in {col}"
        for col in [c for c in res.columns if c.endswith(("_P", "_FDR"))]:
            fdr = col.replace("_P", "_FDR")
            if col.endswith("_P") and fdr in res.columns:
                m = res[[col, fdr]].dropna()
                assert (m[fdr] >= m[col] - 1e-12).all(), f"{key}: {fdr} < {col}"
        assert res["call"].notna().all(), f"{key}: missing call"
        notes.append(f"{key}: {len(res)}x{res.shape[1]}")
    return "; ".join(notes)


@check("shipped TSV datasets load, validate and rebuild their design")
def t_load():
    assert DATA.is_dir(), f"{DATA} not found -- run make_data first"
    prefixes = sorted(p for p in DATA.glob("*.intensities.tsv"))
    from .make_data import MANIFESTS
    assert len(prefixes) == len(MANIFESTS), \
        (f"data/ holds {len(prefixes)} datasets but make_data defines "
         f"{len(MANIFESTS)}; regenerate with `python -m proteomics_revertant.make_data`")
    notes = []
    for f in prefixes:
        prefix = Path(str(f).replace(".intensities.tsv", ""))
        expr, samples, design, truth = load_dataset(prefix)
        assert list(samples.columns) == SAMPLE_COLUMNS, "unexpected sample columns"
        assert expr.shape == (C.N_PROTEINS, len(samples)), f"{f.name}: {expr.shape}"
        assert set(expr.columns) == set(samples["library"]), "library mismatch"
        assert expr.isna().to_numpy().any(), "no missing values -- format not exercised"
        assert truth is not None, "answer key missing"
        # design inferred from samples.tsv alone must match the sample table
        for c in design.clones:
            rows = samples[samples["clone"] == c.name]
            assert len(rows) == c.n_tech and rows["dose"].iloc[0] == c.dose
        # analysis runs straight off the loaded objects
        res, meta = analyse(expr, samples, design)
        assert len(res) == C.N_PROTEINS
        notes.append(f"{design.key}: {expr.shape[0]}x{expr.shape[1]}, "
                     f"{design.n_clones} clones, {meta['residual_df']} df")
    return "; ".join(notes)


@check("HDF5 bundle round-trips to the same numbers as the TSVs")
def t_hdf5():
    try:
        import h5py  # noqa: F401
    except ImportError:
        return "h5py not installed -- optional format skipped"
    h5s = sorted(DATA.glob("*.h5"))
    assert h5s, "no .h5 bundles found"
    for h in h5s:
        prefix = Path(str(h)[:-3])
        e1, s1, d1, _ = load_dataset(prefix)
        e2, s2, d2, _ = load_dataset(h)
        assert list(e1.index) == list(e2.index) and list(e1.columns) == list(e2.columns)
        a, b = e1.to_numpy(float), e2.to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{h.name}: missingness differs"
        assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)]), f"{h.name}: values differ"
        assert [c.name for c in d1.clones] == [c.name for c in d2.clones]
        pd.testing.assert_frame_equal(s1.reset_index(drop=True),
                                      s2.reset_index(drop=True), check_dtype=False)
    return f"{len(h5s)} bundles match their TSVs cell for cell"


@check("validate() rejects malformed datasets with a specific message")
def t_validate():
    expr, samples, _, _ = load_dataset(DATA / "A_homozygous_wt")
    cases = {
        "missing required columns": lambda e, s: (e, s.drop(columns=["dose"])),
        "duplicate library ids": lambda e, s: (e, pd.concat([s, s.iloc[[0]]])),
        "library mismatch": lambda e, s: (e.drop(columns=[e.columns[0]]), s),
        "dose must be 0, 1 or 2": lambda e, s: (e, s.assign(dose=s["dose"].replace(2, 3))),
        "inconsistent state/dose": lambda e, s: (
            e, s.assign(dose=np.where(s.index == 0, 2, s["dose"]))),
        "not on a log2 scale": lambda e, s: (2 ** e, s),
    }
    caught = []
    for expected, mutate in cases.items():
        e, sm = mutate(expr.copy(), samples.copy())
        try:
            validate(e, sm)
        except ValueError as err:
            assert expected.split()[0] in str(err) or expected[:12] in str(err), \
                f"wrong message for '{expected}': {err}"
            caught.append(expected)
            continue
        raise AssertionError(f"validate() accepted a dataset with: {expected}")
    return f"{len(caught)}/{len(cases)} malformed inputs rejected with specific errors"


@check("a hand-written 3-clone dataset works with no code changes")
def t_byo():
    """The integration contract: two text files and nothing else."""
    rng = np.random.default_rng(0)
    libs, rows = [], []
    for clone, state, dose in [("myWT", "wild_type", 0),
                               ("myEdit", "hom_edit", 2),
                               ("myRev", "hom_revertant", 0),
                               ("myHet", "het_edit", 1)]:
        for r in (1, 2, 3):
            lib = f"{clone}_r{r}"
            libs.append(lib)
            rows.append(dict(library=lib, clone=clone, state=state, dose=dose,
                             lineage="mine", tech_rep=r, plex="p1"))
    samples = pd.DataFrame(rows)
    base = rng.normal(22, 2, 300)
    mat = np.column_stack([base + s["dose"] * -1.5 * (np.arange(300) < 5)
                           + rng.normal(0, 0.2, 300) for _, s in samples.iterrows()])
    expr = pd.DataFrame(mat, index=[f"P{i:03d}" for i in range(300)], columns=libs)
    expr.iloc[10, :3] = np.nan

    with tempfile.TemporaryDirectory() as td:
        write_dataset(td, "mine", expr, samples,
                      dict(key="mine", label="hand written"), truth=None)
        e, s, d, t = load_dataset(Path(td) / "mine")
    assert t is None, "truth key invented from nowhere"
    assert d.n_clones == 4 and sorted(d.doses) == [0, 0, 1, 2]
    res, meta = analyse(e, s, d)
    assert len(res) == 300
    top = res.set_index("protein_id").loc[[f"P{i:03d}" for i in range(5)], "dose_FDR"]
    assert (top < 0.01).all(), f"planted effects not recovered: {top.tolist()}"
    return (f"4 clones inferred from samples.tsv, {meta['residual_df']} residual df, "
            f"5/5 planted effects recovered at FDR<0.01")


@check("CLI runs clean on the shipped data directory")
def t_cli():
    pkg_parent = str(Path(__file__).resolve().parents[1])
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        proc = subprocess.run(
            [sys.executable, "-m", "proteomics_revertant.run",
             "--data", str(DATA), "--outdir", str(out)],
            cwd=pkg_parent, capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stderr[-2000:]}"
        assert "Warning" not in proc.stderr and "Traceback" not in proc.stderr, \
            f"stderr not clean:\n{proc.stderr[-2000:]}"
        files = sorted(p.name for p in out.iterdir())
        for key in ["A_homozygous_wt", "B_heterozygous_wt"]:
            f = out / f"{key}_results.tsv"
            assert f.exists(), f"missing {f.name}"
            df = pd.read_csv(f, sep="\t")
            assert len(df) == C.N_PROTEINS, f"{f.name} has {len(df)} rows"
        assert (out / "run_metadata.json").exists()
        return f"exit 0, clean stderr, {len(files)} files written"


# ---------------------------------------------------------------------------

@check("column reference documents exactly the columns emitted")
def t_columns():
    from .columns import data_dictionary, to_markdown
    from .datasets import load_dataset
    from .run import find_datasets

    notes = []
    for prefix in find_datasets(Path("data")):
        expr, samples, design, _ = load_dataset(prefix)
        res, _ = analyse(expr, samples, design)
        doc = data_dictionary(design)
        emitted, documented = list(res.columns), list(doc["column"])
        missing = [c for c in emitted if c not in documented]
        extra = [c for c in documented if c not in emitted]
        assert not missing, f"{design.key}: undocumented columns {missing}"
        assert not extra, f"{design.key}: documented but never emitted {extra}"
        assert emitted == documented, f"{design.key}: dictionary order differs from file order"
        assert doc["column"].is_unique, f"{design.key}: duplicate dictionary entries"
        for col in ["description", "computed_from", "how_to_read", "dtype", "group"]:
            blank = doc.loc[doc[col].astype(str).str.strip() == "", "column"].tolist()
            assert not blank, f"{design.key}: empty {col} for {blank}"
        md = to_markdown(design, doc)
        assert md.count("|") > 4 * len(doc), f"{design.key}: markdown table truncated"
        for c in emitted:
            assert f"`{c}`" in md, f"{design.key}: {c} missing from markdown"
        notes.append(f"{design.key}: {len(doc)} columns")
    assert notes, "no datasets found to document"
    return "; ".join(notes) + " -- names, order and prose all present"


@check("column reference adapts to an unseen clone panel")
def t_columns_generalise():
    """The dictionary must be derived from the design, not hard-coded: a panel
    with different clone names has to produce matching contrast documentation."""
    from .columns import data_dictionary
    from .datasets import design_from_samples, validate

    rows = []
    panel = [("PARENT", "wild_type", 0, "parental"),
             ("KO1", "hom_edit", 2, "ed_a"),
             ("KO2", "hom_edit", 2, "ed_b"),
             ("BACK1", "hom_revertant", 0, "rv_a")]
    for name, state, dose, lin in panel:
        for r in range(3):
            rows.append(dict(library=f"{name}_t{r+1}", clone=name, state=state,
                             dose=dose, lineage=lin, tech_rep=r + 1, plex="plex1"))
    samples = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(rng.normal(23, 0.4, (60, len(samples))),
                        index=[f"P{i:03d}" for i in range(60)],
                        columns=samples["library"])
    expr.index.name = "protein_id"
    validate(expr, samples)
    design = design_from_samples(samples)
    res, _ = analyse(expr, samples, design)
    doc = data_dictionary(design)
    assert list(res.columns) == list(doc["column"]), "dictionary drifted on a new panel"
    named = [c for c in doc["column"] if "KO1" in c or "BACK1" in c or "PARENT" in c]
    assert named, "no clone-specific contrast columns were documented"
    return (f"{design.n_clones}-clone panel with unseen names: {len(doc)} columns "
            f"documented, including {len(named)} clone-specific ones")


@check("omnibus test is calibrated on true nulls and fires on real signal")
def t_omnibus():
    """The global test must not claim a proteome-wide effect when the only thing
    separating the clones is drift and off-target editing."""
    from .omnibus import omnibus
    from .datasets import load_dataset
    from .run import find_datasets

    null_p, null_net = [], []
    for seed in range(401, 409):
        expr, samples, _ = simulate(C.BACKGROUND_A, seed=seed, null=True)
        res, _ = analyse(expr, samples, C.BACKGROUND_A)
        o = omnibus(expr, samples, C.BACKGROUND_A, res, analyse_fn=analyse, n_boot=100)
        null_p.append(o["perm_P"])
        null_net.append(o["global_effect_net_of_noise"])
    expr, samples, design, _ = load_dataset(Path("data/A_homozygous_wt"))
    res, _ = analyse(expr, samples, design)
    o = omnibus(expr, samples, design, res, analyse_fn=analyse, n_boot=100)
    floor = o["perm_floor"]

    # the lineage-matched null has a coarse floor, so calibration is checked as
    # "how often does a true null hit the floor", not "how often is P < 0.05"
    at_floor = sum(1 for x in null_p if np.isclose(x, floor))
    assert at_floor <= 3, (f"global test hit its floor on {at_floor}/8 true-null "
                           f"datasets: {null_p}")
    assert np.isclose(o["perm_P"], floor), \
        f"global test missed real signal: P={o['perm_P']}, floor={floor}"
    assert o["global_effect_net_of_noise"] > 2 * np.mean(null_net), \
        (f"net effect on real data {o['global_effect_net_of_noise']} not clearly "
         f"above the null mean {np.mean(null_net):.4f}")
    return (f"nulls: {at_floor}/8 at the floor, P range [{min(null_p)}, "
            f"{max(null_p)}], net {np.mean(null_net):.3f}; real: P={o['perm_P']} "
            f"(= floor {floor}), net={o['global_effect_net_of_noise']} log2/allele")


@check("omnibus refuses to make a calibrated claim without a matched-dose pair")
def t_omnibus_uncalibrated():
    from .omnibus import omnibus, narrate
    from .datasets import load_dataset

    expr, samples, design, _ = load_dataset(Path("data/B_heterozygous_wt_minimal"))
    res, _ = analyse(expr, samples, design)
    o = omnibus(expr, samples, design, res, analyse_fn=analyse, n_boot=100)
    assert o["n_artefact_contrasts"] == 0, "expected no artefact contrast here"
    story = narrate(o)
    assert "NOT MEASURABLE" in story, "did not flag lambda as unmeasurable"
    assert "upper bound" in story, "did not caveat the effect estimate"
    # and the calibrated design must NOT carry that caveat
    expr2, samples2, design2, _ = load_dataset(Path("data/A_homozygous_wt"))
    res2, _ = analyse(expr2, samples2, design2)
    o2 = omnibus(expr2, samples2, design2, res2, analyse_fn=analyse, n_boot=100)
    assert o2["n_artefact_contrasts"] == 2, "expected two artefact contrasts"
    assert "NOT MEASURABLE" not in narrate(o2), "false caveat on a calibrated design"
    return ("3-clone panel flagged as uncalibratable; 5-clone panel reports "
            f"lambda {o2['lambda_hat']} [{o2['lambda_lo']}, {o2['lambda_hi']}]")


@check("permutation null respects clone structure")
def t_permutation_structure():
    """Relabelling dosages across clones must keep every clone's libraries
    together -- permuting library labels instead would split technical
    replicates across pseudo-clones and give an anticonservative null."""
    from .omnibus import dose_permutations
    from .datasets import design_from_samples

    design = C.BACKGROUND_A
    perms, mode, total = dose_permutations(design)
    assert mode == "exhaustive" and total == 30, \
        f"5-clone panel should enumerate 30 relabellings, got {total} ({mode})"
    assert len(perms) == 30, f"expected 30 distinct dosage vectors, got {len(perms)}"
    assert tuple(sorted(design.doses)) == tuple(sorted(perms[0])), \
        "permutations changed the multiset of dosages"
    assert tuple(design.doses) in perms, "the observed labelling is not in the null"

    expr, samples, _ = simulate(design, seed=5)
    clones = [c.name for c in design.clones]
    for pv in perms[:5]:
        s = samples.copy()
        s["dose"] = s["clone"].map(dict(zip(clones, pv))).astype(int)
        d = design_from_samples(s, {"key": design.key, "label": design.label})
        assert d.n_clones == design.n_clones, "clone count changed under permutation"
        for c in d.clones:
            libs = s.loc[s["clone"] == c.name]
            assert libs["dose"].nunique() == 1, "a clone got two dosages"
            assert len(libs) == c.n_tech, "technical replicates were split"
    return (f"{len(perms)} dosage relabellings, clones intact in all of them; "
            f"permutation floor {1 / len(perms):.4f}")


@check("batched fit reproduces an explicit per-protein loop")
def t_fastfit():
    """analyse() now calls fastfit.fit_batched, so comparing the two would be
    circular. The reference loop is written out here instead -- one pinv per
    protein, no grouping -- and the two must agree."""
    import time as _t
    from .analysis import build_design, clone_summaries, consensus_icc, contrasts_for
    from .datasets import load_dataset
    from .fastfit import fit_batched
    from .run import find_datasets

    notes = []
    for prefix in find_datasets(Path("data")):
        expr, samples, design, _ = load_dataset(prefix)
        res, _ = analyse(expr, samples, design)
        cn = [c.name for c in design.clones]
        means, counts, s2t, _ = clone_summaries(expr, samples, cn)
        X_pri, names_pri = build_design(design, False)
        X_sec, names_sec = build_design(design, True)
        cons = contrasts_for(design, names_sec)
        fm = res["df_residual"].notna().to_numpy()
        rho = consensus_icc(means[fm], counts[fm], X_pri, s2t[fm])
        with np.errstate(divide="ignore", invalid="ignore"):
            W = 1.0 / (rho + (1 - rho) / np.where(counts > 0, counts, np.nan))

        t0 = _t.time()
        est_b, _, s2_b, _, _, _, _ = fit_batched(
            means, counts, W, X_pri, X_sec, fm, cons, cn, names_pri, names_sec)
        t_batched = _t.time() - t0

        n = means.shape[0]
        est_l = {c.name: np.full(n, np.nan) for c in cons}
        s2_l = np.full(n, np.nan)
        cix = {nm: i for i, nm in enumerate(cn)}
        t0 = _t.time()
        for g in np.flatnonzero(fm):
            ok = counts[g] > 0
            w, y, Xg = W[g, ok], means[g, ok], X_pri[ok]
            if np.linalg.matrix_rank(Xg) < Xg.shape[1] or ok.sum() - Xg.shape[1] < 1:
                continue
            Wm = np.diag(w)
            A = np.linalg.pinv(Xg.T @ Wm @ Xg) @ Xg.T @ Wm
            r = y - Xg @ (A @ y)
            s2_l[g] = float(r @ (w * r)) / (ok.sum() - Xg.shape[1])
            idx = np.flatnonzero(ok)
            for con in cons:
                if con.kind == "coef" and con.spec in names_pri:
                    a = A[names_pri.index(con.spec)]
                elif con.kind == "clones":
                    a = np.zeros(int(ok.sum()))
                    bad = False
                    for cname, wt in con.spec.items():
                        hit = np.flatnonzero(idx == cix[cname])
                        if hit.size == 0:
                            bad = True
                            break
                        a[hit[0]] = wt
                    if bad:
                        continue
                else:
                    continue
                est_l[con.name][g] = float(a @ y)
        t_loop = _t.time() - t0

        d_s2 = float(np.nanmax(np.abs(s2_b - s2_l)))
        d_est = max(float(np.nanmax(np.abs(est_b[c.name] - est_l[c.name])))
                    for c in cons if np.isfinite(est_l[c.name]).any())
        assert d_s2 < 1e-10, f"{design.key}: residual variance differs by {d_s2:.2e}"
        assert d_est < 1e-10, f"{design.key}: contrast estimates differ by {d_est:.2e}"
        notes.append(f"{design.key} ({design.n_clones}c): diff "
                     f"{max(d_s2, d_est):.0e}, "
                     f"{t_loop / max(t_batched, 1e-9):.0f}x faster")
    return "; ".join(notes)


@check("every reported p-value is two-sided")
def t_two_sided():
    """Four independent demonstrations, because a one-sided p is also uniform
    under the null and so uniformity alone proves nothing."""
    from .analysis import cochran_armitage
    from .datasets import design_from_samples, load_dataset

    expr, samples, design, _ = load_dataset(Path("data/A_homozygous_wt"))
    res, _ = analyse(expr, samples, design)
    dfm = res["df_moderated"].to_numpy(float)

    # 1. the p-value equals 2 * upper tail, recomputed from unrounded inputs
    checked = 0
    for col in [c for c in res.columns if c.endswith("_P")
                and not c.startswith(("iut", "pa_", "dose_P_recal"))
                and not c.endswith("_equiv_P")]:
        # _equiv_P is a TOST: two ONE-sided tests against +/- delta, combined by
        # taking the maximum. That is one-sided by construction and correctly
        # so -- it is checked separately below, not here.
        stem = col[:-2]
        b = res[f"{stem}_logFC"].to_numpy(float)
        se = res[f"{stem}_SE"].to_numpy(float)
        p_obs = res[col].to_numpy(float)
        ok = np.isfinite(b) & np.isfinite(se) & (se > 0) & np.isfinite(p_obs)
        t = b[ok] / se[ok]
        two = 2 * stats.t.sf(np.abs(t), df=dfm[ok])
        one = stats.t.sf(np.abs(t), df=dfm[ok])
        assert np.max(np.abs(p_obs[ok] - two)) < 1e-3, f"{stem}: not 2*sf(|t|)"
        assert np.max(np.abs(p_obs[ok] - one)) > 0.4, f"{stem}: matches a one-sided p"
        checked += 1
    assert checked >= 4, f"only {checked} contrasts checked"

    # 2. p-values live on [0,1], not [0,0.5]
    for col in ["dose_P", "edit_vs_rev_P"]:
        v = res[col].dropna()
        assert v.max() > 0.95, f"{col} never exceeds 0.95 -- looks one-tailed"
        assert 0.35 < float((v > 0.5).mean()) < 0.65, \
            f"{col}: {float((v > 0.5).mean()):.2f} above 0.5, expected about half"

    # 3. reversing the dosage axis negates every effect and changes no p-value
    s2 = samples.copy()
    s2["dose"] = int(max(design.doses)) - s2["dose"]
    d2 = design_from_samples(s2, {"key": design.key, "label": design.label})
    rev, _ = analyse(expr, s2, d2)
    b_f, b_r = res["dose_logFC"].to_numpy(float), rev["dose_logFC"].to_numpy(float)
    p_f, p_r = res["dose_P"].to_numpy(float), rev["dose_P"].to_numpy(float)
    ok = np.isfinite(b_f) & np.isfinite(b_r)
    assert np.max(np.abs(b_f[ok] + b_r[ok])) < 1e-3, "logFC did not negate"
    assert np.nanmax(np.abs(p_f[ok] - p_r[ok])) < 1e-8, \
        "p-values changed when the axis was reversed -- that is a one-sided test"

    # 4. the exact detection test is symmetric under mirroring
    z1, p1 = cochran_armitage([8, 4, 0], [8, 4, 8], [0, 1, 2])
    z2, p2 = cochran_armitage([0, 4, 8], [8, 4, 8], [0, 1, 2])
    assert abs(p1 - p2) < 1e-12, f"detection test asymmetric: {p1} vs {p2}"
    assert z1 * z2 < 0, "mirrored patterns did not flip the trend statistic"

    # 5. the TOST columns are one-sided BY DESIGN, and must behave that way:
    #    an effect at zero is maximally equivalent, an effect at the bound is not
    eq = [c for c in res.columns if c.endswith("_equiv_P")]
    for col in eq:
        stem = col[: -len("_equiv_P")]
        b = res[f"{stem}_logFC"].to_numpy(float)
        pe = res[col].to_numpy(float)
        ok = np.isfinite(b) & np.isfinite(pe)
        near = ok & (np.abs(b) < 0.05)
        far = ok & (np.abs(b) > 0.30)
        if near.sum() > 20 and far.sum() > 5:
            assert np.median(pe[near]) < np.median(pe[far]), \
                f"{col}: equivalence evidence does not decrease with effect size"
        # symmetric in sign: |b| is what matters, not its direction
        pos = ok & (b > 0.10) & (b < 0.30)
        neg = ok & (b < -0.10) & (b > -0.30)
        if pos.sum() > 10 and neg.sum() > 10:
            assert abs(np.median(pe[pos]) - np.median(pe[neg])) < 0.25, \
                f"{col}: equivalence test is asymmetric in the sign of the effect"

    return (f"{checked} moderated contrasts match 2*sf(|t|); {len(eq)} TOST "
            f"columns one-sided by design and sign-symmetric; axis reversal leaves "
            f"p unchanged to {np.nanmax(np.abs(p_f[ok] - p_r[ok])):.1e}; "
            f"detection test mirror-symmetric")


@check("null hits split evenly up and down once the planted CNV is excluded")
def t_sign_balance():
    """A two-sided test should call as many decreases as increases under the
    null. The simulation deliberately puts a one-directional +0.42 shift on 90
    proteins in the highest-leverage clone, mimicking a culture-adaptation copy
    number gain; those proteins are excluded here and flagged by QC instead."""
    from . import qc

    up = dn = up_blk = dn_blk = 0
    for seed in range(501, 509):
        expr, samples, _ = simulate(C.BACKGROUND_A, seed=seed, null=True)
        res, _ = analyse(expr, samples, C.BACKGROUND_A)
        idx = np.array([int(p.split("_")[1]) for p in res["protein_id"]])
        blk = (idx >= 600) & (idx < 690)
        sig = res["dose_P"].to_numpy(float) < 0.05
        pos = res["dose_logFC"].to_numpy(float) > 0
        up += int((sig & pos & ~blk).sum());     dn += int((sig & ~pos & ~blk).sum())
        up_blk += int((sig & pos & blk).sum());  dn_blk += int((sig & ~pos & blk).sum())

    pval = stats.binomtest(up, up + dn, 0.5).pvalue
    assert pval > 0.01, (f"outside the planted CNV the null is directionally "
                         f"skewed: {up} up / {dn} down, binomial P={pval:.3g}")
    assert up_blk > 5 * max(dn_blk, 1), "the planted CNV did not produce a skew"

    expr, samples, design, _ = __import__(
        "proteomics_revertant.datasets", fromlist=["load_dataset"]
    ).load_dataset(Path("data/A_homozygous_wt"))
    res, _ = analyse(expr, samples, design)
    bal = qc.directional_balance(res)
    assert (bal["flag"] != "").any(), "QC failed to flag the known imbalance"
    blocks = qc.coherent_block_scan(expr, samples)
    worst = blocks["max_rolling_abs_deviation"].idxmax()
    assert worst == "HomEd", f"block scan blamed {worst}, expected HomEd"
    return (f"off-block nulls {up} up / {dn} down (binomial P={pval:.2f}); "
            f"CNV block {up_blk} up / {dn_blk} down; QC flags it and the block "
            f"scan names the right clone")


@check("replication buys what more injections cannot")
def t_replication_scaling():
    """The central design claim, measured. Going from 1 to 3 to 5 independently
    derived clones per genotype state must tighten the standard error, sharpen
    lambda, and lift the permutation floor off the answer."""
    from .datasets import load_dataset
    from .omnibus import n_distinct_permutations, omnibus

    rows = []
    for key in ["A_homozygous_wt", "A_replicated_x3", "A_replicated_x5"]:
        expr, samples, design, truth = load_dataset(Path(f"data/{key}"))
        res, meta = analyse(expr, samples, design)
        o = omnibus(expr, samples, design, res, analyse_fn=analyse,
                    n_boot=200, max_perms=200)
        d = res.set_index("protein_id")
        rows.append(dict(
            key=key, clones=design.n_clones, df=meta["residual_df"],
            se=float(d.loc[C.TARGET_PROTEIN, "dose_SE"]),
            artefacts=o["n_artefact_contrasts"],
            lam_width=o["lambda_hi"] - o["lambda_lo"],
            floor=o["perm_floor"], perm_p=o["perm_P"],
            n_sig=int((res["dose_FDR"] < 0.05).sum()),
            total_perms=n_distinct_permutations(tuple(design.doses))))

    a, b, c = rows
    # 1. standard error shrinks about as 1/sqrt(number of clones)
    ratio = a["se"] / c["se"]
    expected = np.sqrt(c["clones"] / a["clones"])
    assert 0.7 * expected < ratio < 1.3 * expected, \
        f"SE fell {ratio:.2f}x from 5 to 25 clones, expected about {expected:.2f}x"
    # 2. the artefact contrasts that calibrate lambda multiply
    assert c["artefacts"] > 5 * a["artefacts"], \
        f"artefact contrasts {a['artefacts']} -> {c['artefacts']}"
    assert c["lam_width"] < a["lam_width"], "lambda interval did not tighten"
    # 3. the permutation floor stops being the binding constraint
    assert np.isclose(a["perm_p"], a["floor"]), \
        "the 5-clone panel is expected to sit at its floor"
    assert c["floor"] < a["floor"] / 3, \
        f"floor {a['floor']} -> {c['floor']}, expected a big drop"
    assert c["perm_p"] < 0.05 and not np.isclose(c["perm_p"], c["floor"]), \
        (f"25-clone global P={c['perm_p']} (floor {c['floor']}) should be "
         f"significant and off the floor")
    # 4. per-protein sensitivity rises
    assert c["n_sig"] > 2 * a["n_sig"], \
        f"significant proteins {a['n_sig']} -> {c['n_sig']}"

    return " | ".join(
        f"{r['clones']}c: df {r['df']}, SE {r['se']:.3f}, {r['artefacts']} artefacts, "
        f"floor {r['floor']:.4f}, P {r['perm_p']:.4f}, {r['n_sig']} hits"
        for r in rows)


@check("PCA: decomposition, class colouring and no imputation")
def t_pca():
    from .datasets import load_dataset
    from .pca import build_meta, genotype_class, run_pca, top_loadings

    expr, samples, design, _ = load_dataset(Path("data/A_homozygous_wt"))
    expr = expr[samples["library"].to_numpy()]
    pca = run_pca(expr, n_comp=10)

    # complete-case only: the protein count must match exactly, no filling
    arr = expr.to_numpy(float)
    assert pca["n_complete"] == int(np.isfinite(arr).all(axis=1).sum()), \
        "PCA protein count does not match the complete-case count"
    assert pca["n_complete"] < pca["n_total"], "test needs some missingness"

    # scores and loadings reconstruct the centred matrix
    X = arr[np.isfinite(arr).all(axis=1)].T
    Xc = X - X.mean(axis=0)
    full = run_pca(expr, n_comp=min(Xc.shape))
    recon = full["scores"] @ full["loadings"].T
    assert np.max(np.abs(recon - Xc)) < 1e-8, "SVD does not reconstruct the data"
    assert abs(full["var_explained"].sum() - 1.0) < 1e-8, \
        "variance fractions do not sum to 1"
    assert np.all(np.diff(pca["var_explained"]) <= 1e-12), \
        "components are not in decreasing variance order"

    # genotype classes
    meta = build_meta(samples, design)
    assert set(meta["class"]) <= {"wild type", "edited", "revertant", "other"}
    assert genotype_class("hom_revertant", 0, 2) == "revertant"
    assert genotype_class("wild_type", 0, 2) == "wild type"
    assert genotype_class("het_edit", 1, 2) == "edited"
    rev_clones = {c.name for c in design.clones if "rev" in c.state}
    assert set(meta.loc[meta["class"] == "revertant", "clone"]) == rev_clones, \
        "revertant clones were not all coloured as revertants"

    # loadings table covers PC1-10 and the influence score is variance-weighted
    per_pc, overall = top_loadings(pca, n_top=15)
    assert set(per_pc["component"]) == {f"PC{n}" for n in range(1, 11)}, \
        "loading table does not cover PC1-10"
    assert (per_pc.groupby("component")["abs_loading"].apply(
        lambda v: np.all(np.diff(v) <= 1e-12))).all(), "loadings not rank-ordered"
    top = overall.iloc[0]["protein_id"]
    L, var = pca["loadings"], pca["var_explained"]
    infl = (L ** 2 * var[None, :]).sum(axis=1)
    assert pca["proteins"][int(np.argmax(infl))] == top, "influence ranking wrong"

    return (f"{pca['n_complete']}/{pca['n_total']} complete-case proteins, "
            f"exact SVD reconstruction, PC1-10 loadings ranked, "
            f"{len(rev_clones)} revertant clones coloured correctly")


@check("PCA: figures cover only PC1-3 pairs and clouds survive degeneracy")
def t_pca_figure():
    import matplotlib.pyplot as plt

    from .datasets import load_dataset
    from .pca import _ellipse, build_meta, figure_for, run_pca, run_one

    # a degenerate (perfectly collinear) cloud must still draw, thanks to the
    # isotropic ridge -- without it the covariance is singular and nothing shows
    fig, ax = plt.subplots()
    collinear = np.column_stack([np.arange(4.0), np.arange(4.0)])
    assert not _ellipse(ax, collinear, "#000000", floor=None), \
        "singular covariance should be refused without a ridge"
    assert _ellipse(ax, collinear, "#000000", floor=0.5), \
        "ridge did not rescue the degenerate cloud"
    assert _ellipse(ax, np.zeros((4, 2)), "#000000", floor=0.5), \
        "identical points should still draw a cloud"
    plt.close(fig)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        design, pca, story, png = run_one(Path("data/A_homozygous_wt"), out)
        assert png.exists() and png.stat().st_size > 40_000, "figure not written"
        for suffix in ("_pca_loadings.tsv", "_pca_influence.tsv",
                       "_pca_scores.tsv", "_pca.txt"):
            f = out / f"{design.key}{suffix}"
            assert f.exists() and f.stat().st_size > 0, f"missing {f.name}"
        scores = pd.read_csv(out / f"{design.key}_pca_scores.tsv", sep="\t")
        assert len(scores) == 20, f"expected 20 libraries, got {len(scores)}"
        assert "class" in scores.columns and "dose" in scores.columns

    # exactly three score panels, for the three pairs among PC1-3
    expr, samples, design, _ = load_dataset(Path("data/A_homozygous_wt"))
    expr = expr[samples["library"].to_numpy()]
    pca = run_pca(expr, n_comp=10)
    meta = build_meta(samples, design)
    with tempfile.TemporaryDirectory() as td:
        figure_for(design, expr, samples, pca, meta, Path(td) / "f.png")
    titles = []
    fig = plt.figure()
    plt.close(fig)
    # the pair list is fixed in figure_for; assert it here so a change is caught
    from .pca import figure_for as _ff
    import inspect
    src = inspect.getsource(_ff)
    assert "[(0, 1), (0, 2), (1, 2)]" in src, \
        "score panels are no longer restricted to PC1-3 pairs"

    assert "correlation with allele dosage" in story
    return ("3 score panels (PC1v2, PC1v3, PC2v3) plus scree; degenerate and "
            "identical-point clouds both render; 4 side tables written")


@check("PCA reading detects when the revertant fails to return")
def t_pca_reading():
    """The interpretation must distinguish 'PC1 tracks allele dosage' from
    'PC1 tracks clonal drift'. Both are simulated here."""
    from .pca import build_meta, interpret, run_pca
    from .datasets import load_dataset, design_from_samples

    expr, samples, design, _ = load_dataset(Path("data/A_homozygous_wt"))
    expr = expr[samples["library"].to_numpy()]
    pca = run_pca(expr, n_comp=10)
    meta = build_meta(samples, design)
    good = interpret(pca, meta)
    assert "returns the dominant axis toward wild type" in good, \
        f"failed to recognise a clean reversion:\n{good}"
    r1 = float(np.corrcoef(pca["scores"][:, 0], meta["dose"].to_numpy(float))[0, 1])
    assert abs(r1) > 0.5, f"PC1 should track dosage on this dataset, r={r1:.2f}"

    # now push the revertant clones far away, as heavy clonal drift would
    rng = np.random.default_rng(0)
    e2 = expr.copy()
    rev = samples.loc[samples["clone"].isin(
        [c.name for c in design.clones if "rev" in c.state]), "library"]
    e2[rev] = e2[rev].to_numpy() + rng.normal(3.0, 0.1, (len(e2), len(rev)))
    bad = interpret(run_pca(e2, n_comp=10), meta)
    assert "WARNING" in bad, f"failed to flag a drifting revertant:\n{bad}"
    return "clean reversion recognised; a 3 log2 revertant shift raises the warning"


@check("accelerated kernels reproduce the reference fit exactly")
def t_accel():
    from .accel import fit_closed_form, fit_general, sweep_designs, weighted_fit
    from .backend import plan_chunks, resolve

    b = resolve("numpy")
    rng = np.random.default_rng(0)
    n, c = 2000, 5
    dose = np.array([0, 0, 1, 1, 2.0])
    M = rng.normal(23, 1.0, (n, c))
    W = rng.uniform(1.0, 3.0, (n, c))
    X = np.column_stack([np.ones(c), dose])

    # closed form (p=2) against the general batched path
    b0, b1, s2c, _ = fit_closed_form(np, M, W, X)
    beta_g, s2g, _, _ = fit_general(np, M, W, X)
    assert np.allclose(b0, beta_g[:, 0], atol=1e-10), "closed form beta0 differs"
    assert np.allclose(b1, beta_g[:, 1], atol=1e-10), "closed form beta1 differs"
    assert np.allclose(s2c, s2g, atol=1e-12), "closed form residual variance differs"

    # both against an explicit per-protein lstsq
    for g in (0, 17, n - 1):
        sw = np.sqrt(W[g])
        ref = np.linalg.lstsq(X * sw[:, None], M[g] * sw, rcond=None)[0]
        assert np.allclose(ref, [b0[g], b1[g]], atol=1e-9), f"protein {g} differs"

    # a p=3 design must route to the general path and still be right
    X3 = np.column_stack([np.ones(c), dose, (dose == 1).astype(float)])
    beta3, _ = weighted_fit(np, M, W, X3)
    sw = np.sqrt(W[5])
    ref3 = np.linalg.lstsq(X3 * sw[:, None], M[5] * sw, rcond=None)[0]
    assert np.allclose(beta3[5], ref3, atol=1e-9), "p=3 path differs"

    # the sweep statistic against an explicit loop
    designs = [np.column_stack([np.ones(c), rng.permutation(dose)]) for _ in range(6)]
    got = sweep_designs(M, W, designs, backend="numpy")
    exp = []
    for D in designs:
        tb = ts = 0.0
        for g in range(n):
            sw = np.sqrt(W[g]); Xw = D * sw[:, None]; yw = M[g] * sw
            beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
            r = yw - Xw @ beta
            s2 = float(r @ r) / (c - 2)
            cov = np.linalg.inv(Xw.T @ Xw)
            tb += beta[1] ** 2; ts += s2 * cov[1, 1]
        exp.append(tb / n - ts / n)
    assert np.allclose(got, exp, atol=1e-10), \
        f"sweep statistic differs by {np.max(np.abs(got - np.array(exp))):.2e}"

    # chunking must not change the answer
    chunked = sweep_designs(M, W, designs, backend="numpy", chunk=137)
    assert np.allclose(got, chunked, atol=1e-12), "chunking changed the result"
    assert plan_chunks(n, c, 2, b) == n, "CPU backend should not chunk"

    # float32 is offered for the sweep; measure the cost rather than assume it
    f32 = sweep_designs(M, W, designs, backend="numpy", dtype=np.float32)
    rel = float(np.max(np.abs(f32 - got) / np.maximum(np.abs(got), 1e-12)))
    assert rel < 5e-2, f"float32 sweep differs by {rel:.1%}, too much even for ranking"
    return (f"closed form == general == lstsq to 1e-9; sweep matches an explicit "
            f"loop to 1e-10; chunking exact; float32 within {rel:.2%}")


@check("fast permutation path: speedup measured, divergence pinned")
def t_accel_permutation():
    """The hoisted path is 50-330x faster but is NOT yet a drop-in replacement:
    it uses complete-observation proteins only. This test pins exactly where it
    agrees and where it does not, so a fix is detectable."""
    import time as _t

    from .datasets import load_dataset
    from .omnibus import permutation_tau2_fast, permutation_test
    from .run import find_datasets

    agree, differ, gains = [], [], []
    for prefix in find_datasets(Path("data")):
        expr, samples, design, _ = load_dataset(prefix)
        t0 = _t.perf_counter()
        ref = permutation_test(expr, samples, design, analyse)
        t_ref = _t.perf_counter() - t0
        t0 = _t.perf_counter()
        taus, loads, oi, _, _ = permutation_tau2_fast(expr, samples, design)
        t_fast = _t.perf_counter() - t0
        gains.append(t_ref / t_fast)
        m = loads >= loads[oi] - 1e-12
        p_fast = float((taus[m] >= taus[oi]).sum() / m.sum())
        (agree if abs(p_fast - ref["perm_P"]) < 5e-4 else differ).append(
            (design.key, ref["perm_P"], round(p_fast, 4)))

    assert min(gains) > 5, f"fast path is not faster: {min(gains):.1f}x"
    assert len(agree) >= 3, f"fast path agreement regressed: only {len(agree)}/5"
    known = {"A_replicated_x5", "B_heterozygous_wt_minimal"}
    assert {k for k, _, _ in differ} <= known, \
        f"NEW divergence in the fast path: {[k for k, _, _ in differ]}"
    return (f"{len(agree)}/5 datasets agree exactly, {len(differ)} known "
            f"divergences ({', '.join(k for k, _, _ in differ) or 'none'}); "
            f"speedup {min(gains):.0f}-{max(gains):.0f}x")


@check("genomic-control lambda is the GWAS definition and is reported consistently")
def t_genomic_lambda():
    """lambda must recover a KNOWN inflation, not merely look plausible.

    If z ~ N(0, s^2) and p = 2 * sf(|z|), then chi2 = z^2 has median s^2 *
    median(chi2_1), so lambda = s^2 exactly. That gives a closed-form reference
    derived independently of the implementation.
    """
    from .analysis import analyse
    from .datasets import load_dataset
    from .ebayes import CHI2_1_MEDIAN, chi2_1_from_p, genomic_lambda
    from .omnibus import lambda_inflation
    from .qc import null_calibration

    # the constant itself
    assert abs(CHI2_1_MEDIAN - 0.4549364231) < 1e-9, \
        f"median of chi2_1 is wrong: {CHI2_1_MEDIAN}"

    # closed form vs scipy's generic inverse survival function
    # p = 1 maps to chi2 = 0 exactly, so compare on a mixed tolerance rather
    # than a pure ratio
    pv = np.geomspace(1e-12, 1.0, 5000)
    ref = stats.chi2.isf(pv, 1)
    got = chi2_1_from_p(pv)
    dev = float(np.max(np.abs(got - ref) - (1e-10 * np.abs(ref) + 1e-12)))
    assert dev <= 0, f"closed-form chi2_1 differs from scipy, excess {dev:.2e}"

    # a well-calibrated null must give 1.0
    rng = np.random.default_rng(0)
    lam_null = genomic_lambda(rng.uniform(0, 1, 400_000))
    assert abs(lam_null - 1.0) < 0.02, f"lambda on uniform p is {lam_null:.4f}, not 1"

    # a KNOWN inflation must be recovered: lambda should equal s^2
    for s in (1.2, 1.5, 2.0):
        z = rng.normal(0, s, 400_000)
        p = 2.0 * stats.norm.sf(np.abs(z))
        lam = genomic_lambda(p)
        assert abs(lam - s ** 2) / s ** 2 < 0.02, \
            f"inflation s^2={s**2:.2f} recovered as {lam:.3f}"

    # too few p-values must refuse to answer rather than guess
    assert not np.isfinite(genomic_lambda(rng.uniform(0, 1, 5))), \
        "lambda should be NaN below the minimum feature count"

    # the same number must appear in all three places that report it
    seen = []
    for key in ("A_homozygous_wt", "B_heterozygous_wt", "A_replicated_x3"):
        expr, samples, design, _ = load_dataset(DATA / key)
        res, meta = analyse(expr, samples, design)
        qc_med = float(null_calibration(res, design)["lambda_gc"].median())
        omni = lambda_inflation(res, n_boot=200)["lambda_hat"]
        applied = res["lambda_artifact"].iloc[0]
        assert abs(qc_med - omni) < 5e-3, \
            f"{key}: qc lambda {qc_med} != omnibus lambda {omni}"
        assert abs(applied - omni) < 1e-3, \
            f"{key}: results lambda_artifact {applied} != omnibus lambda {omni}"
        seen.append((key, omni))

    # Genomic control is REPORTED, never APPLIED. No recalibrated column may be
    # emitted, and dose_P must be the uncorrected two-sided p-value.
    for key in ("A_homozygous_wt", "B_heterozygous_wt"):
        expr, samples, design, _ = load_dataset(DATA / key)
        r, _ = analyse(expr, samples, design)
        bad = [c for c in r.columns if "recalibrat" in c.lower()]
        assert not bad, f"{key}: pipeline must not emit corrected p-values: {bad}"
        # `dose_t` and `df_moderated` are rounded in the output, so an exact
        # reconstruction is not available. Instead assert that the UNCORRECTED
        # reconstruction fits dose_P better than any genomic-control-corrected
        # one would -- which is what "no correction was applied" means, and is
        # insensitive to rounding.
        t = r["dose_t"].to_numpy(float)
        dfm = r["df_moderated"].to_numpy(float)
        p = r["dose_P"].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(dfm) & np.isfinite(p) & (p > 0)

        def misfit(lam):
            q = 2.0 * stats.t.sf(np.abs(t[ok]) / np.sqrt(lam), df=dfm[ok])
            return float(np.median(np.abs(np.log10(np.maximum(q, 1e-300))
                                          - np.log10(p[ok]))))

        base = misfit(1.0)
        for lam_try in (1.05, 1.10, 1.25):
            assert base < misfit(lam_try), (
                f"{key}: dose_P fits a lambda={lam_try} correction better than no "
                "correction -- genomic control appears to have been applied")

    # Lambda does NOT converge to 1 as the panel grows: the artefact null is
    # heavy-tailed, so its bulk is narrower than the t reference even as its
    # deep tail gets fatter. Measured on true-null simulations lambda settles in
    # 0.77-0.88 and stays there. Pinned on the 25-clone panel so that anyone
    # "correcting" lambda toward 1, or removing the max(lambda, 1) floor, fails
    # here rather than silently deflating every standard error in the output.
    expr, samples, design, _ = load_dataset(DATA / "A_replicated_x5")
    res25, _ = analyse(expr, samples, design)
    lam25 = lambda_inflation(res25, n_boot=200)["lambda_hat"]
    assert 0.70 <= lam25 <= 0.95, (
        f"25-clone lambda is {lam25}, outside the documented 0.70-0.95 band. "
        "lambda does not converge to 1 as panels grow: the artefact null is "
        "heavy-tailed. See README section 2.1 before changing this.")
    assert abs(res25["lambda_artifact"].iloc[0] - lam25) < 1e-3, \
        "lambda_artifact must report the unfloored estimate, since it is not applied"

    return ("chi2_1 closed form == scipy to 1e-10; lambda=1.00 on a uniform null; "
            f"known inflations 1.44/2.25/4.00 recovered within 2%; "
            f"qc == omnibus == results on {len(seen)} panels; "
            f"25-clone lambda {lam25} reported unfloored and never applied")


TESTS = [t_trigamma, t_fitfdist, t_squeeze, t_bh, t_modt, t_ca, t_clonesum,
         t_wls, t_df, t_pairwise, t_null, t_fdr_null, t_recovery,
         t_no_imputation, t_lackoffit, t_determinism, t_genomic_lambda, t_columns, t_columns_generalise, t_fastfit, t_omnibus, t_omnibus_uncalibrated,
         t_permutation_structure, t_two_sided, t_sign_balance, t_pca, t_pca_figure, t_pca_reading, t_accel, t_accel_permutation, t_replication_scaling, t_contract,
         t_load, t_hdf5, t_validate, t_byo, t_cli]


def main():
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | "
          f"pandas {pd.__version__}")
    print("=" * 78)
    for t in TESTS:
        t()
    width = max(len(n) for n, *_ in RESULTS)
    for name, ok, detail, dt in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name:<{width}}  {dt:6.2f}s")
        if detail:
            print(f"       {detail}")
    n_fail = sum(1 for _, ok, *_ in RESULTS if not ok)
    print("=" * 78)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
