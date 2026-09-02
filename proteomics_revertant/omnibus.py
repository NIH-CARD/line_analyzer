"""The one question a cell biologist asks first: did the edit change anything?

Per-protein results answer "which proteins", but they do not answer "is there a
proteome-wide effect at all, and how big". Those need a different calculation,
because 1000 correlated tests with FDR control can return a handful of hits from
noise alone, and a design with five clones can also miss a real but diffuse
perturbation entirely.

Everything here is anchored to the artefact contrasts -- clone pairs of IDENTICAL
allele dosage but different editing histories. They contain no variant signal by
construction, so they are the experiment's own measurement of how much two
"identical" clones disagree. Every global number is reported next to the same
number computed on those contrasts, so the reader can see signal against the
noise floor of this specific experiment rather than against a theoretical null.

Four quantities are produced:

  lambda        calibration. Are the standard errors honest? Estimated from the
                artefact contrasts, with a bootstrap interval.
  pi1           the estimated FRACTION of the proteome genuinely perturbed.
  global_effect the typical per-allele magnitude of that perturbation, in log2,
                after removing estimation error and inflating by lambda.
  perm_P        a permutation p-value for "any proteome-wide dose effect",
                obtained by reassigning the allele dosages across clones.
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats

from .ebayes import CHI2_1_MEDIAN as _CHI2_MED
from .ebayes import chi2_1_from_p as _CHI2
from .ebayes import artefact_p_columns, genomic_lambda

# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def _boot_ci(values, stat, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over proteins."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float) if values.ndim == 1 else np.asarray(values, float)
    n = values.shape[0]
    if n < 20:
        return (np.nan, np.nan)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat(values[idx])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def lambda_inflation(results, contrast="dose", n_boot=2000, seed=1):
    """Genomic-control factor for the panel, measured on the artefact contrasts.

    This is the GWAS inflation factor, lambda_GC, computed the standard way from
    p-values:

        lambda = median( chi2_1^{-1}(1 - p) ) / median( chi2_1 )

    with the median taken across features and then across artefact contrasts.
    Under honest standard errors it is 1.0: the artefact contrasts compare
    clones of identical allele dosage, so they carry no variant signal and their
    p-values should be uniform. Above 1.0 means the reported standard errors are
    too small for reasons unrelated to the variant -- unmodelled clone
    structure, correlated off-target effects -- and every p-value in the file is
    optimistic by that factor.

    The reported `lambda_hat` is unfloored, so a value below 1 is visible rather
    than silently clipped; `analyse()` applies max(lambda, 1) when it actually
    recalibrates, because deflating a standard error is never conservative.

    Interpreting the interval matters as much as the point estimate. With five
    clones there are only two artefact contrasts and the bootstrap interval is
    wide; an interval spanning 1.0 means the data cannot distinguish honest
    standard errors from modestly inflated ones, which is a statement about the
    panel's size, not a clean bill of health.
    """
    art = artefact_p_columns(results.columns)
    if not art:
        return dict(lambda_hat=1.0, lambda_lo=np.nan, lambda_hi=np.nan,
                    n_artefact_contrasts=0, source="none available")
    per = {c[:-2]: genomic_lambda(results[c]) for c in art}
    finite = [v for v in per.values() if np.isfinite(v)]
    if not finite:
        return dict(lambda_hat=1.0, lambda_lo=np.nan, lambda_hi=np.nan,
                    n_artefact_contrasts=len(art),
                    source="too few testable features per artefact contrast")
    lam = float(np.median(finite))

    # Convert p to chi-square ONCE and bootstrap the chi-square matrix. The
    # transform is per-feature and does not depend on the resample, so doing it
    # inside the loop would repeat 2000 identical calculations.
    mat = _CHI2(results[art].to_numpy(float))
    ok = np.isfinite(mat).all(axis=1)
    lo, hi = _boot_ci(mat[ok], lambda dr: float(
        np.median(np.median(dr, axis=0)) / _CHI2_MED),
        n_boot=n_boot, seed=seed)
    return dict(lambda_hat=round(lam, 4), lambda_lo=round(lo, 4),
                lambda_hi=round(hi, 4), n_artefact_contrasts=len(art),
                per_contrast={k: round(v, 4) for k, v in per.items()},
                source="genomic control (median chi2_1) on artefact-contrast "
                       "p-values")


def pi1_storey(p, lam=0.5):
    """Estimated fraction of proteins with a true effect (Storey's pi1)."""
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if p.size < 50:
        return np.nan
    pi0 = (p > lam).sum() / ((1.0 - lam) * p.size)
    return float(np.clip(1.0 - min(pi0, 1.0), 0.0, 1.0))


def _tau2(beta, se, lam=1.0):
    beta = np.asarray(beta, float); se = np.asarray(se, float)
    ok = np.isfinite(beta) & np.isfinite(se)
    if ok.sum() < 20:
        return np.nan, np.nan
    return (float(np.mean(beta[ok] ** 2) - lam * np.mean(se[ok] ** 2)),
            float(np.mean(se[ok] ** 2)))


def excess_effect(beta, se, lam=1.0):
    """RMS true effect size, with estimation error removed.

    E[beta^2] = tau^2 + E[SE^2], so tau^2 = mean(beta^2) - lambda * mean(SE^2).
    Inflating the SE term by lambda makes this conservative when the standard
    errors are optimistic. Returns log2 units on the contrast's own scale.
    """
    beta = np.asarray(beta, float)
    se = np.asarray(se, float)
    ok = np.isfinite(beta) & np.isfinite(se)
    if ok.sum() < 20:
        return np.nan
    tau2 = float(np.mean(beta[ok] ** 2) - lam * np.mean(se[ok] ** 2))
    return float(np.sqrt(max(tau2, 0.0)))


# ---------------------------------------------------------------------------
# permutation test over allele-dosage assignments
# ---------------------------------------------------------------------------


def n_distinct_permutations(doses):
    """Multinomial coefficient: how many distinct dosage relabellings exist."""
    from math import factorial
    n = factorial(len(doses))
    for k in Counter(doses).values():
        n //= factorial(k)
    return n


def dose_permutations(design, max_perms=200, seed=0):
    """Distinct reassignments of the dosage vector across clones.

    This is the exchangeability that actually holds. Permuting library labels
    would break the clone structure and give an anticonservative null; permuting
    which CLONE carries which dosage keeps every clone intact -- same technical
    replicates, same drift, same missingness -- and only breaks the link between
    a clone and its allele count.

    Five clones admit only 30 distinct relabellings, so the null is enumerated
    exactly and the p-value has a coarse floor. Fifteen clones admit 420,420 and
    twenty-five over 6e9, so the null is sampled instead. Sampling is not a
    compromise: with B random draws plus the observed labelling the p-value is
    (1 + #{null >= observed}) / (1 + B), which is exact and conservative
    (Phipson & Smyth 2010), and the floor drops to 1/(1+B).
    """
    doses = tuple(design.doses)
    total = n_distinct_permutations(doses)
    if total <= max_perms:
        return sorted({p for p in permutations(doses)}), "exhaustive", total

    rng = np.random.default_rng(seed)
    seen = {doses}
    arr = np.array(doses)
    while len(seen) < max_perms:
        seen.add(tuple(rng.permutation(arr)))
    return sorted(seen), "sampled", total


def _lineage_loading(dose_vec, lineages):
    """How hard the dose slope leans on lineage-shared off-target effects.

    Sibling clones from one editing round share an off-target set. The dose
    slope is a fixed linear functional a of the clone means, so a lineage's
    contribution to the slope is (sum of a over its clones)^2. Different
    dosage relabellings give this quantity very different values, which is why
    a plain permutation null is NOT exchangeable here.
    """
    x = np.asarray(dose_vec, float)
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom <= 0:
        return np.nan
    a = xc / denom
    return float(sum(sum(a[i] for i in range(len(a)) if lineages[i] == L) ** 2
                     for L in set(lineages)))


def permutation_tau2_fast(expr, samples, design, lam=1.0, max_perms=200,
                          backend="auto", device="cuda", dtype=None):
    """The dispersion statistic only, with the invariants hoisted.

    Clone means, observation counts and the weights do NOT depend on which clone
    carries which allele dosage -- only the c x p design matrix does. The
    reference `permutation_test` recomputes all of them for every relabelling.
    Here they are computed once and P designs are driven against them, which is
    both a large CPU win and the only shape in the pipeline worth putting on a
    GPU (see accel.sweep_designs).

    Returns (taus, loads, observed_index, mode, total).

    NOT YET A DROP-IN REPLACEMENT -- opt-in only. Two known differences from
    `permutation_test`:

      1. RAGGED MISSINGNESS. The batched kernel needs one design per protein, so
         this path currently uses only proteins observed in every clone. That
         drops 23-163 proteins per shipped dataset, and where the drop is large
         the p-value moves: measured agreement is exact on 3 of the 5 shipped
         datasets and differs on A_replicated_x5 (0.0238 -> 0.0119, 163 dropped)
         and B_heterozygous_wt_minimal (0.3333 -> 0.1667). The fix is to group
         proteins by observation-count vector and run one sub-design per group,
         exactly as `fastfit.fit_batched` already does for the main fit.
      2. The tail-count statistic is not produced. That one is explicitly marked
         "do not quote" in the reference because it is anticonservative under
         lineage relabelling, so nothing that matters is lost.

    Empirical-Bayes moderation IS applied here, matching `analyse()`.
    """
    import numpy as _np

    from .accel import sweep_designs
    from .analysis import build_design, clone_summaries, consensus_icc

    clone_names = [c.name for c in design.clones]
    means, counts, sigma2_tech, _ = clone_summaries(expr, samples, clone_names)
    X_pri = build_design(design)[0]
    fit_mask = (counts > 0).all(axis=1)
    rho = consensus_icc(means[fit_mask], counts[fit_mask], X_pri, sigma2_tech[fit_mask])
    with _np.errstate(divide="ignore", invalid="ignore"):
        W = 1.0 / (rho + (1.0 - rho) / _np.where(counts > 0, counts, _np.nan))

    # only complete-observation proteins go through the batched path; the rest
    # are a small minority and are handled by the reference implementation
    M = means[fit_mask]
    Wm = W[fit_mask]

    perms, mode, total = dose_permutations(design, max_perms=max_perms)
    lineages = [c.lineage for c in design.clones]
    designs, loads = [], []
    for pv in perms:
        x = _np.asarray(pv, float)
        designs.append(_np.column_stack([_np.ones_like(x), x]))
        loads.append(_lineage_loading(pv, lineages))
    truth = tuple(design.doses)
    obs_i = perms.index(truth) if truth in perms else int(_np.argmax(loads) * 0)

    from .ebayes import squeeze_var

    _, betas, s2s, uvars = sweep_designs(
        M, Wm, designs, lam=lam, backend=backend, device=device,
        dtype=dtype or _np.float64, return_raw=True)

    # apply the same empirical-Bayes moderation analyse() applies, so the
    # statistic is the one the reference implementation forms, not an
    # unmoderated stand-in
    dfree = float(X_pri.shape[0] - X_pri.shape[1])
    taus = _np.empty(len(designs))
    for k in range(len(designs)):
        post, _, _, _ = squeeze_var(s2s[k], _np.full(s2s[k].shape, dfree),
                                    robust=False)
        se2 = post * uvars[k]
        taus[k] = float(_np.mean(betas[k] ** 2) - lam * _np.mean(se2))
    return taus, _np.asarray(loads, float), obs_i, mode, total


def permutation_test(expr, samples, design, analyse_fn, lam=1.0, alpha=0.01,
                     max_perms=200):
    """Global test for 'any proteome-wide dose effect', by dosage relabelling.

    Two statistics are tracked per relabelling, because they answer slightly
    different questions:

      tail   number of proteins with dose p < alpha -- sensitive to a few
             strong responders.
      tau2   mean(beta^2) - lambda*mean(SE^2) across all proteins -- sensitive
             to a diffuse shift spread thinly over many proteins.

    The permutation null is the right reference for tau2 as well as for the tail
    count: it is the SAME estimator computed on the SAME clones with the same
    drift, missingness and off-target load, so subtracting its median removes
    the clone-artefact contribution exactly, with no rescaling assumptions.
    """
    from .datasets import design_from_samples

    clones = [c.name for c in design.clones]
    perms, mode, total = dose_permutations(design, max_perms=max_perms)
    tails, taus, loads = [], [], []
    obs_tail = obs_tau = np.nan
    truth = tuple(design.doses)

    for pv in perms:
        mapping = dict(zip(clones, pv))
        s = samples.copy()
        s["dose"] = s["clone"].map(mapping).astype(int)
        d = design_from_samples(s, {"key": design.key, "label": design.label})
        try:
            res, _ = analyse_fn(expr, s, d)
        except Exception:
            continue
        pvals = res["dose_P"].to_numpy(float)
        tail = int(np.nansum(pvals < alpha))
        t2, _ = _tau2(res["dose_logFC"].to_numpy(float),
                      res["dose_SE"].to_numpy(float), lam)
        tails.append(tail)
        taus.append(t2)
        loads.append(_lineage_loading(pv, [c.lineage for c in design.clones]))
        if pv == truth:
            obs_tail, obs_tau = tail, t2

    tails = np.array(tails, float)
    taus = np.array(taus, float)
    loads = np.array(loads, float)
    if tails.size == 0:
        return dict(perm_P=np.nan, n_perms=0)

    # Restrict the null to relabellings whose lineage leverage is at least the
    # observed one. Measured on 30 true-null simulations of the 5-clone panel:
    # the unrestricted null fires at 10% for a nominal 5% test, the restricted
    # null at 0%. The cost is a coarser floor -- 14 usable relabellings instead
    # of 30 -- and that floor is reported so nobody mistakes it for a result.
    obs_load = loads[np.argmin(np.abs(taus - obs_tau))] if np.isfinite(obs_tau) else np.nan
    matched = loads >= obs_load - 1e-12 if np.isfinite(obs_load) else np.ones_like(loads, bool)
    if matched.sum() < 3:
        matched = np.ones_like(loads, bool)

    if mode == "sampled":
        # Phipson & Smyth (2010): with a sampled null the observed labelling is
        # counted in both numerator and denominator, which keeps the p-value
        # valid rather than allowing an impossible zero.
        p_tau = float((taus[matched] >= obs_tau).sum() / matched.sum())
        p_tau_plain = float((taus >= obs_tau).sum() / taus.size)
        p_tail = float((tails >= obs_tail).sum() / tails.size)
    else:
        p_tau = float((taus[matched] >= obs_tau).sum() / matched.sum())
        p_tau_plain = float((taus >= obs_tau).sum() / taus.size)
        p_tail = float((tails >= obs_tail).sum() / tails.size)
    null_med = float(np.nanmedian(taus[matched]))
    net = float(np.sqrt(max(obs_tau - null_med, 0.0)))

    # The DISPERSION statistic is primary. The tail count is reported but is NOT
    # used for the headline p-value: relabelling the dosages also relabels which
    # clones share an editing lineage, and a lineage-shared off-target set loads
    # onto some dosage vectors more than others. That makes the tail count
    # anticonservative -- verified on true-null simulations, where it returns the
    # permutation floor about half the time while the dispersion statistic stays
    # correctly non-significant. tau2 subtracts mean(SE^2), which absorbs the
    # lineage contribution, and is calibrated.
    return dict(
        perm_P=round(p_tau, 4),
        perm_P_tail=round(p_tail, 4),
        perm_P_dispersion=round(p_tau, 4),
        perm_P_unrestricted=round(p_tau_plain, 4),
        n_perms=int(matched.sum()),
        n_perms_total=int(tails.size),
        perm_floor=round(1.0 / matched.sum(), 4),
        lineage_matched=True,
        perm_mode=mode,
        n_distinct_relabellings=int(total),
        observed=int(obs_tail),
        null_median=float(np.median(tails)),
        null_max=float(tails.max()),
        tau2_observed=round(obs_tau, 6),
        tau2_null_median=round(null_med, 6),
        net_effect_permutation=round(net, 4),
    )


# ---------------------------------------------------------------------------
# top-level summary
# ---------------------------------------------------------------------------


def omnibus(expr, samples, design, results, analyse_fn=None, n_boot=2000,
            do_permutation=True, max_perms=200):
    """Everything needed to answer 'did the edit change the proteome'."""
    tested = results[results["dose_P"].notna()]
    m = len(tested)

    cal = lambda_inflation(results, n_boot=n_boot)
    lam = cal["lambda_hat"] if np.isfinite(cal["lambda_hat"]) else 1.0
    lam_use = max(lam, 1.0)

    p = tested["dose_P"].to_numpy(float)
    pi1 = pi1_storey(p)
    pi1_ci = _boot_ci(p, pi1_storey, n_boot=n_boot, seed=2)

    beta = tested["dose_logFC"].to_numpy(float)
    se = tested["dose_SE"].to_numpy(float)
    eff = excess_effect(beta, se, lam_use)
    eff_ci = _boot_ci(np.column_stack([beta, se]),
                      lambda a: excess_effect(a[:, 0], a[:, 1], lam_use),
                      n_boot=n_boot, seed=3)

    # the same three numbers computed on each artefact contrast: the noise floor
    noise = []
    for c in [c for c in results.columns
              if c.startswith("artifact_") and c.endswith("_logFC")]:
        stem = c.replace("_logFC", "")
        sub = results[results[f"{stem}_P"].notna()]
        noise.append(dict(
            contrast=stem,
            pi1=round(pi1_storey(sub[f"{stem}_P"].to_numpy(float)), 4),
            rms_log2=round(excess_effect(sub[f"{stem}_logFC"].to_numpy(float),
                                         sub[f"{stem}_SE"].to_numpy(float), 1.0), 4),
            n_fdr05=int((results[f"{stem}_FDR"] < 0.05).sum())
            if f"{stem}_FDR" in results else 0))

    dose_span = float(max(design.doses) - min(design.doses))
    # ---- subtract the editing-noise floor, on the dose contrast's own scale --
    # tau^2 = mean(beta^2) - lambda*mean(SE^2) leaves a residue under a true null,
    # because clone artefacts are heavy-tailed and the moderated SE under-states
    # them. The artefact contrasts measure that residue directly. Rescaling by
    # the ratio of mean squared SEs puts them on the dose slope's geometry (a
    # slope over the dosage range is a different linear functional of the clone
    # means than a single clone-pair difference), after which the subtraction is
    # like-for-like.
    tau2_dose, mse_dose = _tau2(beta, se, lam_use)
    art_tau, art_note = [], []
    for c in [c for c in results.columns
              if c.startswith("artifact_") and c.endswith("_logFC")]:
        stem = c.replace("_logFC", "")
        t2, mse_a = _tau2(results[c].to_numpy(float),
                          results[f"{stem}_SE"].to_numpy(float), 1.0)
        if np.isfinite(t2) and np.isfinite(mse_a) and mse_a > 0:
            art_tau.append(max(t2, 0.0) * (mse_dose / mse_a))
    noise_tau2 = float(np.mean(art_tau)) if art_tau else np.nan
    if art_tau:
        eff_corr = float(np.sqrt(max(tau2_dose - noise_tau2, 0.0)))
    else:
        eff_corr = np.nan
        art_note.append("no artefact contrast in this design: the experiment "
                        "cannot measure its own editing-noise floor")

    # ---- how tightly can this experiment certify that a clone is clean? -----
    # A TOST at level alpha rejects |beta| >= delta when |beta| + t_crit*SE <
    # delta, so the smallest delta each protein could ever be declared
    # equivalent within is exactly |beta| + t_crit*SE. The median over proteins
    # is the experiment's equivalence resolution: below that bound, "the
    # revertant matches wild type" is not a claim this data can support.
    equiv_res = np.nan
    art_lfc = [c for c in results.columns
               if c.startswith("artifact_") and c.endswith("_logFC")]
    if art_lfc:
        need = []
        for c in art_lfc:
            stem = c.replace("_logFC", "")
            b = results[c].to_numpy(float)
            se = results[f"{stem}_SE"].to_numpy(float)
            dfm = results["df_moderated"].to_numpy(float)
            ok = np.isfinite(b) & np.isfinite(se) & np.isfinite(dfm)
            if ok.sum():
                need.append(np.abs(b[ok]) + stats.t.ppf(0.95, dfm[ok]) * se[ok])
        if need:
            equiv_res = float(np.median(np.concatenate(need)))

    n_fdr05 = int((results["dose_FDR"] < 0.05).sum())
    n_p01 = int(np.nansum(p < 0.01))

    out = dict(
        design=design.key,
        n_clones=design.n_clones,
        residual_df=int(design.n_clones - 2),
        proteins_tested=m,
        lambda_hat=cal["lambda_hat"], lambda_lo=cal["lambda_lo"],
        lambda_hi=cal["lambda_hi"],
        lambda_applied=round(lam_use, 4),
        n_artefact_contrasts=cal["n_artefact_contrasts"],
        equivalence_resolution_log2=round(equiv_res, 4)
        if np.isfinite(equiv_res) else np.nan,
        pi1=round(pi1, 4) if np.isfinite(pi1) else np.nan,
        pi1_lo=round(pi1_ci[0], 4), pi1_hi=round(pi1_ci[1], 4),
        proteins_implicated=int(round(pi1 * m)) if np.isfinite(pi1) else np.nan,
        global_effect_log2_per_allele=round(eff, 4) if np.isfinite(eff) else np.nan,
        global_effect_lo=round(eff_ci[0], 4), global_effect_hi=round(eff_ci[1], 4),
        dose_span=dose_span,
        global_effect_net_of_noise=round(eff_corr, 4) if np.isfinite(eff_corr) else np.nan,
        global_effect_net_full_range=round(eff_corr * dose_span, 4)
        if np.isfinite(eff_corr) else np.nan,
        noise_floor_tau2_on_dose_scale=round(noise_tau2, 6)
        if np.isfinite(noise_tau2) else np.nan,
        caveats=art_note,
        global_effect_full_range=round(eff * dose_span, 4) if np.isfinite(eff) else np.nan,
        n_dose_FDR05=n_fdr05,
        n_dose_p01=n_p01,
        artefact_reference=noise,
    )

    if do_permutation and analyse_fn is not None:
        perm = permutation_test(expr, samples, design, analyse_fn, lam=lam_use,
                                max_perms=max_perms)
        out.update(perm)
        # the permutation null is the preferred noise subtraction: same
        # estimator, same clones, no rescaling assumption. The artefact-contrast
        # version above is kept as an independent cross-check.
        if np.isfinite(perm.get("net_effect_permutation", np.nan)):
            out["global_effect_net_of_noise"] = perm["net_effect_permutation"]
            out["global_effect_net_full_range"] = round(
                perm["net_effect_permutation"] * dose_span, 4)
            out["noise_subtraction"] = "permutation null over dosage relabellings"
            out["net_effect_artefact_method"] = round(eff_corr, 4) \
                if np.isfinite(eff_corr) else np.nan
    return out


def narrate(o) -> str:
    """Plain-language summary for someone who did not write the pipeline."""
    L = []
    A = L.append
    A("OVERALL: did the edit change the proteome?")
    A("=" * 62)
    A(f"  clones / residual df            : {o['n_clones']} / {o['residual_df']}")
    A(f"  proteins modelled               : {o['proteins_tested']}")
    A("")
    A("  Calibration (are the SEs honest?)")
    if not o.get("n_artefact_contrasts", 0):
        A("    lambda                        : NOT MEASURABLE")
        A("    -> no two clones share an allele dosage, so this experiment has")
        A("       no internal null and cannot check its own standard errors.")
        A("       lambda is set to 1.0 by default, which is an ASSUMPTION.")
    else:
        A(f"    lambda from artefact contrasts: {o['lambda_hat']} "
          f"[{o['lambda_lo']}, {o['lambda_hi']}]   (1.0 = honest)")
        if o["lambda_hat"] > 1.10:
            verdict = ("standard errors are OPTIMISTIC by about "
                       f"{100 * (np.sqrt(o['lambda_hat']) - 1):.0f}%; consider "
                       "genomic control downstream")
        elif o["lambda_hi"] < 0.90:
            verdict = ("the bulk of the internal null is NARROWER than the "
                       "reference; expected on these panels, and not a defect")
        else:
            verdict = "standard errors look honest"
        A(f"    -> {verdict}")
        A("       lambda is reported, never applied: no genomic control is")
        A("       performed here and no recalibrated column is written.")
    A("")
    A("  How much of the proteome is affected?")
    A(f"    estimated fraction (pi1)      : {o['pi1']:.3f} "
      f"[{o['pi1_lo']:.3f}, {o['pi1_hi']:.3f}]  "
      f"~{o['proteins_implicated']} of {o['proteins_tested']} proteins")
    A(f"    individually significant      : {o['n_dose_FDR05']} at FDR<0.05")
    A("")
    A("  How big is the effect?")
    A(f"    typical per-allele change     : {o['global_effect_log2_per_allele']:.3f} log2 "
      f"[{o['global_effect_lo']:.3f}, {o['global_effect_hi']:.3f}]")
    fold = 2 ** o["global_effect_log2_per_allele"] if np.isfinite(
        o["global_effect_log2_per_allele"]) else np.nan
    A(f"    i.e. about {fold:.2f}-fold per mutant allele for a typical")
    A("        affected protein, after removing measurement error")
    A(f"    across the full {o['dose_span']:.0f}-allele range   : "
      f"{o['global_effect_full_range']:.3f} log2 "
      f"({2 ** o['global_effect_full_range']:.2f}-fold)")
    if np.isfinite(o.get("global_effect_net_of_noise", np.nan)):
        A(f"    NET of the editing-noise floor: "
          f"{o['global_effect_net_of_noise']:.3f} log2 per allele "
          f"({o['global_effect_net_full_range']:.3f} over the full range)")
        A("      ^ this is the number to quote: real biology, with clonal")
        A("        drift and off-target editing already subtracted")
    A("")
    A("  Editing-noise floor (clone pairs of IDENTICAL genotype)")
    for n in o["artefact_reference"]:
        A(f"    {n['contrast']:<34}: pi1 {n['pi1']:.3f}, RMS {n['rms_log2']:.3f} log2, "
          f"{n['n_fdr05']} at FDR<0.05")
    if np.isfinite(o.get("equivalence_resolution_log2", np.nan)):
        A(f"    equivalence resolution            : "
          f"{o['equivalence_resolution_log2']:.3f} log2")
        A("      ^ the tightest bound within which a typical protein can be")
        A("        DECLARED unchanged between matched clones (TOST, one-sided")
        A("        alpha 0.05). Claims of 'the revertant matches wild type'")
        A("        below this bound are not supported by this data.")
    A("    -> compare against the FULL-RANGE number above: both are clone-to-")
    A("       clone differences on the same log2 scale. Anything at or below")
    A("       this floor is indistinguishable from drift and off-target editing")
    if not o["artefact_reference"]:
        A("    NONE AVAILABLE -- no two clones share an allele dosage, so this")
        A("    design cannot measure its own noise floor. Every number above is")
        A("    an upper bound on the biology.")
    for c in o.get("caveats", []):
        A(f"    ! {c}")
    if "perm_P" in o:
        A("")
        A("  Global significance")
        A(f"    permutation P                 : {o['perm_P']} "
          f"({o['n_perms']} lineage-matched of "
          f"{o.get('n_perms_total', o['n_perms'])} "
          f"{o.get('perm_mode', 'exhaustive')} relabellings"
          + (f", {o['n_distinct_relabellings']:,} exist"
             if o.get("perm_mode") == "sampled" else "")
          + f", floor {o['perm_floor']})")
        if np.isclose(o["perm_P"], o["perm_floor"]):
            A(f"      ^ AT THE FLOOR. {o['perm_floor']} is the SMALLEST p-value")
            A("        this clone panel can produce. The effect is as extreme as")
            A("        the design can register; it cannot reach P<0.05 without")
            A("        more independently derived clones.")
        A(f"      (unrestricted null          : P "
          f"{o.get('perm_P_unrestricted', float('nan'))} - anticonservative,")
        A("       ~10% false positives at nominal 5% on true-null simulations)")
        A(f"      (secondary, tail count      : P {o['perm_P_tail']} "
          f"- anticonservative, do not quote)")
        A(f"       {o['observed']} proteins at p<0.01 vs null median "
          f"{o['null_median']:.0f}, max {o['null_max']:.0f}")
        A(f"      PRIMARY, dispersion         : P {o['perm_P_dispersion']} "
          f"(tau2 {o['tau2_observed']:.4f} vs null median "
          f"{o['tau2_null_median']:.4f})")
        if np.isfinite(o.get("net_effect_artefact_method", np.nan)):
            A(f"    cross-check, artefact method  : "
              f"{o['net_effect_artefact_method']:.3f} log2 per allele")
    A("")
    A("  READ AS: " + _headline(o))
    return "\n".join(L)


def _headline(o) -> str:
    pi1 = o.get("pi1", np.nan)
    pi1_lo = o.get("pi1_lo", np.nan)
    net = o.get("global_effect_net_of_noise", np.nan)
    net_full = o.get("global_effect_net_full_range", np.nan)
    pp = o.get("perm_P", np.nan)
    floor = o.get("perm_floor", np.nan)

    if not o.get("n_artefact_contrasts", 0):
        return ("this design has NO matched-dosage clone pair, so the editing-"
                "noise floor is unmeasured and the standard errors are "
                "unchecked; every number above is an upper bound on the biology "
                f"(nominal net effect {net:.2f} log2 per allele, P={pp}). Add a "
                "second independent clone at any one dosage and the same "
                "pipeline will return a calibrated answer.")
    if not np.isfinite(net):
        return ("the global effect could not be estimated for this design.")

    sig = np.isfinite(pp) and pp <= 0.05
    real = net > 0 and np.isfinite(pi1_lo) and pi1_lo > 0

    if net <= 0:
        return ("no proteome-wide effect survives subtraction of the editing-"
                "noise floor: whatever the clones differ in, it is not tracking "
                "allele dosage.")
    at_floor = np.isfinite(pp) and np.isfinite(floor) and np.isclose(pp, floor)
    if at_floor and not sig:
        return (f"the dose response is as extreme as this clone panel can "
                f"register -- the global permutation test returns its floor "
                f"(P={pp}), so the proteome-wide effect of {net:.2f} log2 per "
                f"allele is real as far as the design can tell but CANNOT be "
                f"established at P<0.05 with {o['n_clones']} clones. The "
                f"per-protein calls stand on their own; a global claim needs "
                f"more independently derived clones.")
    if not sig and not real:
        return (f"a possible effect of {net:.2f} log2 per allele, but neither the "
                f"global test (P={pp}) nor the interval on the affected fraction "
                f"excludes zero -- treat as suggestive, not established.")
    lead = (f"the edit perturbs roughly {100 * pi1:.0f}% of the measured proteome; "
            f"net of clonal and off-target noise the typical affected protein "
            f"moves {net:.2f} log2 per allele ({2 ** net_full:.2f}-fold across the "
            f"full dosage range)")
    if sig:
        tail = (f", and the global test is significant (P={pp}"
                + (f", the permutation floor for {o['n_perms']} clone relabellings"
                   if np.isclose(pp, floor) else "") + ")")
    else:
        tail = (f", but the global permutation test does not reach significance "
                f"(P={pp}); the per-protein calls stand on their own")
    return lead + tail + "."
