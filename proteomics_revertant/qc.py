"""QC that specifically targets the failure modes of edited-clone experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def missingness_report(expr, samples):
    m = ~np.isfinite(expr.to_numpy(float))
    per_lib = pd.DataFrame({
        "library": expr.columns,
        "pct_missing": np.round(100 * m.mean(axis=0), 2),
    }).merge(samples[["library", "clone", "state", "dose", "plex"]], on="library")
    return per_lib


def missingness_vs_abundance(expr):
    """MNAR check: dropout should rise steeply at low intensity."""
    arr = expr.to_numpy(float)
    mean_int = np.nanmean(arr, axis=1)
    frac_missing = np.mean(~np.isfinite(arr), axis=1)
    ok = np.isfinite(mean_int)
    bins = pd.qcut(pd.Series(mean_int[ok]), 8, duplicates="drop")
    tab = (pd.DataFrame({"bin": bins.values, "frac_missing": frac_missing[ok]})
           .groupby("bin", observed=True)["frac_missing"].agg(["mean", "size"]))
    tab.columns = ["mean_frac_missing", "n_proteins"]
    return tab.reset_index()


def artefact_scale(results, design):
    """Robust SD of the matched-dose / within-state artefact contrasts.

    This is the empirical floor of the experiment: effects smaller than this are
    indistinguishable from clonal drift and off-target editing.
    """
    cols = [c for c in results.columns
            if c.startswith("artifact_") and c.endswith("_logFC")]
    rows = []
    for c in cols:
        v = results[c].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        mad = float(np.median(np.abs(v - np.median(v))))
        rows.append(dict(
            contrast=c.replace("_logFC", ""),
            n=v.size,
            median=round(float(np.median(v)), 4),
            robust_sd=round(1.4826 * mad, 4),
            frac_abs_gt_0p5=round(float(np.mean(np.abs(v) > 0.5)), 4),
        ))
    return pd.DataFrame(rows)


def null_calibration(results, design):
    """Are artefact-contrast statistics over-dispersed relative to N(0,1)?

    If lambda >> 1 the nominal standard errors are too small for reasons that
    have nothing to do with the variant, and the primary contrast should be
    recalibrated (or the clone variance component inflated).
    """
    out = []
    for c in [c for c in results.columns
              if c.startswith("artifact_") and c.endswith("_t")]:
        t = results[c].to_numpy(float)
        t = t[np.isfinite(t)]
        if t.size < 20:
            continue
        mad = float(np.median(np.abs(t - np.median(t))))
        out.append(dict(contrast=c.replace("_t", ""),
                        robust_sd_of_t=round(1.4826 * mad, 3),
                        lambda_gc=round((1.4826 * mad) ** 2, 3)))
    return pd.DataFrame(out)


def clone_correlation(expr, samples):
    """Complete-case correlation between clone mean profiles."""
    clones = samples["clone"].unique()
    prof = {}
    for cl in clones:
        libs = samples.loc[samples["clone"] == cl, "library"]
        prof[cl] = expr[libs].mean(axis=1, skipna=True)
    P = pd.DataFrame(prof).dropna()
    return P.corr(method="pearson").round(4)


def coherent_block_scan(expr, samples, window=50):
    """Crude CNV mimic detector: rolling mean of a clone's deviation from the
    grand mean along the protein index. A sustained shift is a red flag."""
    clones = samples["clone"].unique()
    prof = {}
    for cl in clones:
        libs = samples.loc[samples["clone"] == cl, "library"]
        prof[cl] = expr[libs].mean(axis=1, skipna=True)
    P = pd.DataFrame(prof)
    dev = P.sub(P.mean(axis=1), axis=0)
    roll = dev.rolling(window, center=True, min_periods=window // 2).mean()
    return roll.abs().max().round(4).rename("max_rolling_abs_deviation").to_frame()


def directional_balance(results, alpha=0.05):
    """Are nominally significant hits split evenly up and down?

    Every p-value in the pipeline is two-sided, so under a true null the
    significant hits should be about half up and half down. A strong imbalance
    is not a problem with the test -- it is a signal that ONE clone carries a
    coherent one-directional shift, which is what a culture-adaptation copy
    number change looks like. Pair this with `coherent_block_scan`: if the
    imbalanced proteins also cluster together, that is a CNV, not biology.
    """
    from scipy import stats as _st

    rows = []
    for col in [c for c in results.columns if c.endswith("_P")]:
        stem = col[:-2]
        lfc = f"{stem}_logFC"
        if lfc not in results.columns:
            continue
        sub = results[(results[col] < alpha) & results[lfc].notna()]
        up = int((sub[lfc] > 0).sum())
        dn = int((sub[lfc] < 0).sum())
        if up + dn < 10:
            continue
        rows.append(dict(
            contrast=stem, n_nominal=up + dn, n_up=up, n_down=dn,
            frac_up=round(up / (up + dn), 3),
            binomial_P=float(_st.binomtest(up, up + dn, 0.5).pvalue),
            flag="IMBALANCED - check for a clone-specific coherent shift"
            if _st.binomtest(up, up + dn, 0.5).pvalue < 0.01 else ""))
    return pd.DataFrame(rows)
