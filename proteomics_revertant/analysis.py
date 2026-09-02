"""Differential abundance across an allelic series, without imputation.

Design principles baked in here:

1.  The unit of replication is the CLONE, not the library. Technical replicates
    are collapsed to clone means and the residual df is (clones - parameters).
2.  Nothing is imputed. Proteins absent from a whole genotype state are routed
    to a detection-count test instead of being filled in and t-tested.
3.  Filtering is on presence in counts ("at least 2 observations in at least one
    state"), never on percentage missing across the whole matrix -- the latter
    deletes exactly the on/off proteins a knockout is supposed to produce.
4.  The primary estimand is the per-allele dose slope. Pairwise state contrasts
    are tested against the clone-level residual, which is dominated by the
    matched-dose pairs and is therefore an honest artefact-scale error term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import warnings
from itertools import product

from scipy import stats
from scipy.special import comb

from . import config as C
from .ebayes import (artefact_p_columns, bh, genomic_lambda, moderated_t,
                     squeeze_var)
from .fastfit import fit_batched


# ---------------------------------------------------------------------------
# clone-level reduction
# ---------------------------------------------------------------------------

def clone_summaries(expr: pd.DataFrame, samples: pd.DataFrame, clone_names):
    """Per protein x clone: observation count, mean, within-clone SS and df."""
    n_prot = expr.shape[0]
    n_cl = len(clone_names)
    means = np.full((n_prot, n_cl), np.nan)
    counts = np.zeros((n_prot, n_cl), dtype=int)
    within_ss = np.zeros(n_prot)
    within_df = np.zeros(n_prot)

    for j, cl in enumerate(clone_names):
        libs = samples.loc[samples["clone"] == cl, "library"].values
        block = expr[libs].to_numpy(float)
        obs = np.isfinite(block)
        k = obs.sum(axis=1)
        counts[:, j] = k
        with np.errstate(invalid="ignore"):
            m = np.nansum(np.where(obs, block, 0.0), axis=1) / np.where(k > 0, k, np.nan)
        means[:, j] = m
        dev = np.where(obs, block - m[:, None], 0.0)
        within_ss += np.nansum(dev**2, axis=1)
        within_df += np.maximum(k - 1, 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        sigma2_tech = np.where(within_df > 0, within_ss / np.where(within_df > 0, within_df, np.nan), np.nan)
    return means, counts, sigma2_tech, within_df


def build_design(design, secondary: bool = False):
    dose = np.array(design.doses, float)
    cols = [np.ones_like(dose), dose]
    names = ["intercept", "dose"]
    if secondary:
        het = (dose == 1).astype(float)
        # only estimable if dose-1 clones exist and aren't collinear with dose
        if 0 < het.sum() < len(dose):
            cols.append(het)
            names.append("dominance")
    return np.column_stack(cols), names


def consensus_icc(means, counts, X, sigma2_tech):
    """Genome-wide median intraclass correlation (clone variance / total)."""
    n_prot, n_cl = means.shape
    p = X.shape[1]
    rho = []
    for g in range(n_prot):
        ok = counts[g] > 0
        if ok.sum() <= p or not np.isfinite(sigma2_tech[g]):
            continue
        Xg, yg = X[ok], means[g, ok]
        beta, *_ = np.linalg.lstsq(Xg, yg, rcond=None)
        resid = yg - Xg @ beta
        dfree = ok.sum() - p
        if dfree <= 0:
            continue
        ms_res = float(resid @ resid) / dfree
        inv_n = float(np.mean(1.0 / counts[g, ok]))
        tau2 = max(ms_res - sigma2_tech[g] * inv_n, 0.0)
        denom = tau2 + sigma2_tech[g]
        if denom > 0:
            rho.append(tau2 / denom)
    if not rho:
        return 0.5
    return float(np.clip(np.median(rho), 0.05, 0.95))


# ---------------------------------------------------------------------------
# contrast definitions per design
# ---------------------------------------------------------------------------

@dataclass
class Contrast:
    name: str
    kind: str            # "coef" (from the fitted model) or "clones" (clone means)
    spec: object         # coefficient name, or dict {clone: weight}
    role: str            # primary / corroborating / artefact / secondary


def _groups(design):
    """Split the clone panel into edit / revertant / baseline by dose and name."""
    doses = np.array(design.doses)
    names = [c.name for c in design.clones]
    is_rev = np.array(["rev" in c.state.lower() for c in design.clones])

    edit = [n for n, d, r in zip(names, doses, is_rev) if d == doses.max() and not r]
    if not edit:
        edit = [n for n, d in zip(names, doses) if d == doses.max()]
    rev = [n for n, r in zip(names, is_rev) if r and n not in edit]
    if rev:
        rev_dose = min(d for n, d in zip(names, doses) if n in rev)
        rev = [n for n, d in zip(names, doses) if n in rev and d == rev_dose]
    base = [n for n, r in zip(names, is_rev) if not r and n not in edit]
    if base:
        base_dose = min(d for n, d in zip(names, doses) if n in base)
        base = [n for n, d in zip(names, doses) if n in base and d == base_dose]
    return edit, rev, base


def contrasts_for(design, coef_names):
    """Contrasts derived from the clone panel, not from a hard-coded design key.

    The primary estimand is the dose slope. `edit_vs_rev` and `edit_vs_baseline`
    are the corroborating pair for the intersection-union test. Any dose level
    carrying two or more clones yields an artefact contrast automatically: those
    clones are the same genotype with different editing histories, so their
    difference is pure clonal and off-target noise.
    """
    out = [Contrast("dose", "coef", "dose", "primary")]
    if "dominance" in coef_names:
        out.append(Contrast("dominance", "coef", "dominance", "secondary"))

    edit, rev, base = _groups(design)

    def avg(group, sign):
        return {n: sign / len(group) for n in group}

    if edit and rev:
        spec = {**avg(edit, 1.0), **avg(rev, -1.0)}
        out.append(Contrast("edit_vs_rev", "clones", spec, "corroborating"))
    if edit and base:
        spec = {**avg(edit, 1.0), **avg(base, -1.0)}
        out.append(Contrast("edit_vs_baseline", "clones", spec, "corroborating"))

    # The aggregate revertant-vs-baseline contrast: the comparison the whole
    # design is built around. Its standard error shrinks with the number of
    # clones on each side, unlike the pairwise artefact contrasts below, so it
    # is the one that can actually certify "the reversion restored wild type"
    # via an equivalence test. Added only when replication makes it distinct
    # from a single clone pair.
    if rev and base and (len(rev) > 1 or len(base) > 1):
        spec = {**avg(rev, 1.0), **avg(base, -1.0)}
        out.append(Contrast("revertant_vs_baseline", "clones", spec, "artefact"))

    # matched-dose / within-state pairs: the empirical editing-noise contrasts
    by_dose = {}
    for c in design.clones:
        by_dose.setdefault(c.dose, []).append(c.name)
    for d in sorted(by_dose):
        members = by_dose[d]
        for i in range(len(members) - 1):
            a, b = members[i], members[i + 1]
            out.append(Contrast(f"artifact_{a}_vs_{b}", "clones",
                                {a: 1.0, b: -1.0}, "artefact"))
    return out


def iut_pair(design):
    """The two contrasts whose intersection defines a variant-attributable call."""
    return "edit_vs_rev", "edit_vs_baseline"


# ---------------------------------------------------------------------------
# presence / absence track
# ---------------------------------------------------------------------------

def _ca_stat(det, n, scores, pbar):
    return float(np.sum(scores * (np.asarray(det) - np.asarray(n) * pbar)))


def cochran_armitage(det, n, scores, exact_limit=200000):
    """Trend test on detection counts against allele dose.

    With 12-20 libraries the normal approximation is unreliable and the exact
    permutation null is cheap, so we enumerate it (multivariate hypergeometric
    over which libraries were detected) and fall back to the z approximation
    only if the enumeration would be large.
    """
    det = np.asarray(det, int)
    n = np.asarray(n, int)
    scores = np.asarray(scores, float)
    N, R = int(n.sum()), int(det.sum())
    if N == 0 or R == 0 or R == N:
        return np.nan, np.nan

    pbar = R / N
    T_obs = _ca_stat(det, n, scores, pbar)
    var = pbar * (1 - pbar) * (float(np.sum(n * scores**2)) - float(np.sum(n * scores)) ** 2 / N)
    z = T_obs / np.sqrt(var) if var > 0 else np.nan

    total = comb(N, R, exact=True)
    if total <= exact_limit and len(n) <= 4:
        p_exact = 0.0
        ranges = [range(0, min(ni, R) + 1) for ni in n[:-1]]
        for combo in product(*ranges):
            last = R - sum(combo)
            if last < 0 or last > n[-1]:
                continue
            r = np.array(list(combo) + [last], int)
            ways = 1
            for ni, ri in zip(n, r):
                ways *= comb(ni, ri, exact=True)
            T = _ca_stat(r, n, scores, pbar)
            if abs(T) >= abs(T_obs) - 1e-9:
                p_exact += ways / total
        return z, float(min(p_exact, 1.0))

    if not np.isfinite(z):
        return np.nan, np.nan
    return z, float(2 * stats.norm.sf(abs(z)))


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def analyse(expr: pd.DataFrame, samples: pd.DataFrame, design,
            min_obs: int = C.MIN_OBS_PER_STATE, equiv_delta: float = 0.2):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # a wide clone panel produces ~5 columns per contrast, and pandas warns
    # about repeated inserts; the frame is defragmented once before returning
    warnings.filterwarnings("ignore", message=".*highly fragmented.*")
    proteins = expr.index.to_numpy()
    n_prot = len(proteins)
    clone_names = [c.name for c in design.clones]
    dose = np.array(design.doses, float)
    states = [c.state for c in design.clones]

    means, counts, sigma2_tech, within_df = clone_summaries(expr, samples, clone_names)

    # ---- observation bookkeeping per genotype state ------------------------
    # sort by (dosage, name): sorting on dosage alone leaves states that share a
    # dosage to be ordered by set iteration, which is hash-randomised and made
    # the obs_* column order differ between processes
    state_levels = sorted(set(states), key=lambda s: (
        min(d for d, st in zip(dose, states) if st == s), s))
    state_obs = {}
    for st in state_levels:
        cols = [i for i, s in enumerate(states) if s == st]
        state_obs[st] = counts[:, cols].sum(axis=1)
    state_obs_df = pd.DataFrame(state_obs, index=proteins)

    n_libs = expr.shape[1]
    n_obs_total = np.isfinite(expr.to_numpy(float)).sum(axis=1)

    # ---- filtering / track assignment --------------------------------------
    obs_arr = state_obs_df.to_numpy()
    any_state_ok = (obs_arr >= min_obs).any(axis=1)
    all_states_ok = (obs_arr >= min_obs).all(axis=1)
    a_state_empty = (obs_arr == 0).any(axis=1)

    clones_with_data = (counts > 0).sum(axis=1)
    distinct_dose = np.array([len(set(dose[counts[g] > 0])) for g in range(n_prot)])
    p_primary = build_design(design)[0].shape[1]
    modelable = (clones_with_data >= p_primary + 1) & (distinct_dose >= 2)

    track = np.full(n_prot, "insufficient", dtype=object)
    track[any_state_ok & modelable] = "continuous"
    track[any_state_ok & a_state_empty] = "presence_absence"
    track[any_state_ok & a_state_empty & modelable] = "presence_absence+continuous"

    fit_mask = np.isin(track, ["continuous", "presence_absence+continuous"])

    # ---- weights from the consensus ICC ------------------------------------
    X_pri, names_pri = build_design(design, secondary=False)
    X_sec, names_sec = build_design(design, secondary=True)
    rho = consensus_icc(means[fit_mask], counts[fit_mask], X_pri, sigma2_tech[fit_mask])

    with np.errstate(divide="ignore", invalid="ignore"):
        W = 1.0 / (rho + (1.0 - rho) / np.where(counts > 0, counts, np.nan))

    # ---- per-protein weighted fit -----------------------------------------
    cons = contrasts_for(design, names_sec)
    # The per-protein loop lives in fastfit.fit_batched: proteins sharing an
    # observation-count vector share a design and weights, so one factorisation
    # serves the whole group. Same arithmetic as the loop (tests assert
    # agreement to 1e-15), roughly an order of magnitude faster, which matters
    # because the permutation null in omnibus.py refits everything hundreds of
    # times.
    est, unscaled, s2, df_resid, s2_sec, df_sec, used_secondary = fit_batched(
        means, counts, W, X_pri, X_sec, fit_mask, cons, clone_names,
        names_pri, names_sec)

    # ---- empirical Bayes moderation ---------------------------------------
    # Shrinkage prior from the additive residuals ...
    post_var, s02, d0, d0_vec = squeeze_var(s2, np.nan_to_num(df_resid, nan=0.0),
                                            robust=False)
    # ... but the hypervariability test uses the SATURATED residual. A strongly
    # recessive protein has a large additive residual purely from lack of fit;
    # exempting it from shrinkage on that basis would inflate its standard error
    # and bury a real hit. Only proteins that stay extreme after the dose axis is
    # saturated are genuinely noisy clones-wise, and those keep their own
    # variance instead of borrowing the bulk prior.
    outlier = np.zeros(n_prot, bool)
    if np.isfinite(s2_sec).sum() > 50:
        _, s02s, d0s, _ = squeeze_var(s2_sec, np.nan_to_num(df_sec, nan=0.0), robust=False)
        if np.isfinite(d0s) and s02s > 0:
            with np.errstate(invalid="ignore"):
                tail = stats.f.sf(s2_sec / s02s,
                                  dfn=np.where(df_sec > 0, df_sec, 1), dfd=d0s)
            outlier = np.isfinite(tail) & (tail < 0.01)
    if outlier.any():
        d0_vec = np.where(outlier, 0.0, d0_vec)
        with np.errstate(invalid="ignore", divide="ignore"):
            post_var = np.where(np.isfinite(s2),
                                (d0_vec * s02 + df_resid * np.nan_to_num(s2)) /
                                np.where(d0_vec + df_resid > 0, d0_vec + df_resid, np.nan),
                                np.nan)
    df_total = np.where(np.isfinite(df_resid), df_resid + d0_vec, np.nan)

    out = pd.DataFrame(index=pd.Index(proteins, name="protein_id"))
    out["track"] = track
    out["n_libraries"] = n_libs
    out["n_observed"] = n_obs_total
    out["pct_missing"] = np.round(100.0 * (1 - n_obs_total / n_libs), 2)
    for st in state_levels:
        out[f"obs_{st}"] = state_obs_df[st].values
    out["mean_log2_intensity"] = np.round(np.nanmean(expr.to_numpy(float), axis=1), 4)
    out["sigma2_tech"] = np.round(sigma2_tech, 5)
    out["clone_var_moderated"] = np.round(post_var, 5)
    out["df_residual"] = df_resid
    out["df_moderated"] = np.round(df_total, 2)
    out["variance_outlier"] = np.isfinite(df_resid) & (d0_vec == 0.0)
    out["dominance_estimable"] = used_secondary

    for con in cons:
        se, t, p = moderated_t(est[con.name], unscaled[con.name], post_var, df_total)
        out[f"{con.name}_logFC"] = np.round(est[con.name], 4)
        out[f"{con.name}_SE"] = np.round(se, 4)
        out[f"{con.name}_t"] = np.round(t, 3)
        out[f"{con.name}_P"] = p
        if con.role in ("primary", "secondary", "artefact"):
            out[f"{con.name}_FDR"] = bh(p)

    # ---- equivalence test on the artefact contrasts -----------------------
    # "The revertant is not significantly different from wild type" is absence
    # of evidence, and it gets easier to claim the noisier the experiment is.
    # The positive claim needs two one-sided tests (Schuirmann 1987): reject
    # both |beta| >= +delta and |beta| <= -delta and you have evidence the
    # difference is negligible. delta defaults to 0.2 log2, a ~15% change.
    for con in cons:
        if con.role != "artefact":
            continue
        b = out[f"{con.name}_logFC"].to_numpy(float)
        se = out[f"{con.name}_SE"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_lo = stats.t.sf((b + equiv_delta) / se, df=df_total)
            p_hi = stats.t.cdf((b - equiv_delta) / se, df=df_total)
        p_eq = np.maximum(p_lo, p_hi)
        out[f"{con.name}_equiv_P"] = p_eq
        out[f"{con.name}_equiv_FDR"] = bh(p_eq)

    # ---- empirical-null recalibration (genomic control on artefact contrasts) --
    # The artefact contrasts contain no variant signal by construction: they
    # compare clones of identical allele dosage, so whatever separates them is
    # drift and off-target editing. That makes their p-values a designed null,
    # and the genomic-control factor computed on them a direct check on whether
    # the nominal standard errors are honest. lambda > 1 means they are too
    # small and every p-value in the file is optimistic by that factor.
    #
    # This is the GWAS estimator, median(chi2_1^-1(1-p)) / median(chi2_1), taken
    # over the artefact-contrast p-values -- not over `dose_P`. In a GWAS the
    # null is "most markers do nothing"; here the null is a designed feature of
    # the panel, which is a stronger footing. Using `dose_P` would fold genuine
    # variant signal into the calibration constant and shrink real effects.
    # Selected by `artefact_p_columns`, NOT by contrast role: the role also
    # covers the aggregate revertant-vs-baseline contrast, which re-uses the
    # same clones and would make this lambda disagree with the one omnibus and
    # qc report for the identical run.
    art_p = [out[c].to_numpy(float) for c in artefact_p_columns(out.columns)]
    lam = np.nan
    if art_p:
        lams = [genomic_lambda(p) for p in art_p]
        lams = [v for v in lams if np.isfinite(v)]
        if lams:
            lam = float(np.median(lams))
    out["lambda_artifact"] = round(lam, 4) if np.isfinite(lam) else np.nan

    # REPORTED, NOT APPLIED. The pipeline deliberately does not perform genomic
    # control: no recalibrated p-value column is emitted and `dose_P` is never
    # divided by anything. Correcting is the reader's decision, not the
    # pipeline's, and it is one line downstream:
    #
    #     chi2 = chi2_1_from_p(results["dose_P"]) / lam
    #     p_gc = stats.chi2.sf(chi2, 1)
    #
    # Two reasons the pipeline declines to do it. First, on these panels lambda
    # is usually BELOW 1 (see "lambda below 1 is normal here" in README section
    # 2.1), and
    # dividing by a lambda under 1 inflates every statistic -- the opposite of a
    # correction, and dangerous where the deep tail is already fat. Second, a
    # column named `*_recalibrated` invites quoting without the reader ever
    # deciding whether the correction was warranted. The estimate is reported
    # unfloored so the sign of the departure from 1 stays visible.

    # ---- intersection-union call ------------------------------------------
    c1, c2 = iut_pair(design)
    if f"{c1}_P" in out and f"{c2}_P" in out:
        p1, p2 = out[f"{c1}_P"].to_numpy(), out[f"{c2}_P"].to_numpy()
        b1, b2 = out[f"{c1}_logFC"].to_numpy(), out[f"{c2}_logFC"].to_numpy()
        concord = np.sign(b1) == np.sign(b2)
        p_iut = np.nanmax(np.vstack([p1, p2]), axis=0)
        p_iut = np.where(concord, p_iut, 1.0)
        p_iut = np.where(np.isfinite(p1) & np.isfinite(p2), p_iut, np.nan)
        out["sign_concordant"] = concord
        out["iut_P"] = p_iut
        out["iut_FDR"] = bh(p_iut)

    # ---- monotonicity in dose ---------------------------------------------
    mono = np.full(n_prot, False)
    dose_levels = sorted(set(dose))
    if len(dose_levels) >= 3:
        grp = []
        for d in dose_levels:
            cols = np.flatnonzero(dose == d)
            grp.append(np.nanmean(means[:, cols], axis=1))
        G = np.vstack(grp)
        diffs = np.diff(G, axis=0)
        with np.errstate(invalid="ignore"):
            mono = np.all(diffs >= 0, axis=0) | np.all(diffs <= 0, axis=0)
        mono = np.where(np.isfinite(G).all(axis=0), mono, False)
    out["monotone_in_dose"] = mono

    # ---- presence/absence track -------------------------------------------
    lib_dose = samples["dose"].to_numpy(float)
    detected = np.isfinite(expr.to_numpy(float))
    pa_z = np.full(n_prot, np.nan)
    pa_p = np.full(n_prot, np.nan)
    uniq = np.array(sorted(set(lib_dose)))
    n_by_dose = np.array([(lib_dose == d).sum() for d in uniq])
    pa_rows = np.flatnonzero(np.isin(track, ["presence_absence", "presence_absence+continuous"]))
    for g in pa_rows:
        det = np.array([detected[g, lib_dose == d].sum() for d in uniq])
        z, p = cochran_armitage(det, n_by_dose, uniq)
        pa_z[g], pa_p[g] = z, p
    out["pa_trend_z"] = np.round(pa_z, 3)
    out["pa_trend_P"] = pa_p
    out["pa_trend_FDR"] = bh(pa_p)

    # ---- final call --------------------------------------------------------
    # priority: variant_attributable > presence_absence_hit > dose_trend_only
    call = np.full(n_prot, "not_significant", dtype=object)
    testable = fit_mask | np.isin(track, ["presence_absence", "presence_absence+continuous"])
    call[~testable] = "filtered_no_model"

    dose_sig = np.nan_to_num(out["dose_FDR"].to_numpy(), nan=1.0) < 0.05
    call[dose_sig & testable] = "dose_trend_only"

    pa_sig = np.nan_to_num(out["pa_trend_FDR"].to_numpy(), nan=1.0) < 0.05
    call[pa_sig] = "presence_absence_hit"

    if "iut_FDR" in out:
        iut_sig = (np.nan_to_num(out["iut_FDR"].to_numpy(), nan=1.0) < 0.05) & \
                  out["monotone_in_dose"].to_numpy()
        call[iut_sig] = "variant_attributable"

    out["pa_hit"] = pa_sig
    art_fdr = [out[f"{c.name}_FDR"].to_numpy() for c in cons
               if c.role == "artefact" and f"{c.name}_FDR" in out]
    if art_fdr:
        out["artifact_flag"] = np.any(
            [np.nan_to_num(a, nan=1.0) < 0.05 for a in art_fdr], axis=0)
    out["call"] = call

    meta = dict(
        design=design.key,
        label=design.label,
        n_clones=design.n_clones,
        n_libraries=n_libs,
        residual_df=int(design.n_clones - p_primary),
        consensus_icc=round(rho, 4),
        prior_df_d0=round(float(d0), 3) if np.isfinite(d0) else np.inf,
        prior_var_s02=round(float(s02), 5),
        contrasts=[c.name for c in cons],
        iut_pair=(c1, c2),
        lambda_artifact=round(lam, 4),
        equivalence_delta_log2=equiv_delta,
    )
    return out.copy().reset_index(), meta
