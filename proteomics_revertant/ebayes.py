"""Empirical Bayes variance moderation, ported from limma's squeezeVar.

With five clones and two model parameters there are three residual degrees of
freedom per protein. On its own that is far too thin. Borrowing a variance prior
across all 1000 proteins is what makes the design analysable: the moderated
t-statistic is tested on d + d0 df rather than d.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats


def trigamma_inverse(x: np.ndarray) -> np.ndarray:
    """Solve trigamma(y) = x for y (limma's trigammaInverse)."""
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)

    small, big = x > 1e7, x < 1e-6
    ok = ~(small | big)
    y[small] = 1.0 / np.sqrt(x[small])
    y[big] = 1.0 / x[big]

    if ok.any():
        xi = x[ok]
        yi = 0.5 + 1.0 / xi
        for _ in range(50):
            tri = special.polygamma(1, yi)
            dif = tri * (1.0 - tri / xi) / special.polygamma(2, yi)
            yi = yi + dif
            if np.max(np.abs(dif / yi)) < 1e-8:
                break
        y[ok] = yi
    return y


def fit_f_dist(var: np.ndarray, df1: np.ndarray, trim: float = 0.0):
    """Estimate the scaled-F prior (s0^2, d0) from observed residual variances.

    `trim` winsorises that fraction from each tail of the log-variance before
    taking moments, which keeps a heavy upper tail of artefact-driven variances
    from dragging the prior.
    """
    var = np.asarray(var, float)
    df1 = np.broadcast_to(np.asarray(df1, float), var.shape)

    ok = np.isfinite(var) & (var > 0) & (df1 > 0)
    if ok.sum() < 3:
        return float(np.nanmedian(var[ok])) if ok.any() else 1.0, 0.0

    v, d = var[ok], df1[ok]
    z = np.log(v)
    e = z - special.digamma(d / 2.0) + np.log(d / 2.0)
    if trim > 0 and e.size > 20:
        lo, hi = np.quantile(e, [trim, 1.0 - trim])
        e = np.clip(e, lo, hi)
    emean = e.mean()
    n = e.size
    evar = ((e - emean) ** 2).sum() / (n - 1) - special.polygamma(1, d / 2.0).mean()

    if evar > 0:
        d0 = 2.0 * float(trigamma_inverse(np.array([evar]))[0])
        s02 = float(np.exp(emean + special.digamma(d0 / 2.0) - np.log(d0 / 2.0)))
    else:
        d0 = np.inf
        s02 = float(np.exp(emean))
    return s02, d0


def squeeze_var(var: np.ndarray, df: np.ndarray, robust: bool = True,
                outlier_p: float = 0.01):
    """Shrink per-protein variances toward a scaled-F prior.

    Returns (post_var, s02, d0, per_protein_d0).

    With `robust=True` (limma's `eBayes(robust=TRUE)` in spirit) the prior is
    fitted on trimmed moments and proteins whose residual variance sits in the
    extreme upper tail of the prior predictive F distribution are exempted from
    shrinkage. That matters a lot here: clone artefacts make the variance
    distribution a mixture, and shrinking a genuinely hypervariable protein
    toward the bulk prior shrinks its standard error too, which is
    anti-conservative exactly where you least want it to be.
    """
    var = np.asarray(var, float)
    df = np.broadcast_to(np.asarray(df, float), var.shape).astype(float)
    s02, d0 = fit_f_dist(var, df, trim=0.02 if robust else 0.0)

    d0_vec = np.full(var.shape, d0 if np.isfinite(d0) else 1e6, float)

    if robust and np.isfinite(d0) and s02 > 0:
        ok = np.isfinite(var) & (var > 0) & (df > 0)
        Fstat = np.where(ok, var / s02, np.nan)
        with np.errstate(invalid="ignore"):
            tail = stats.f.sf(Fstat, dfn=np.where(df > 0, df, 1), dfd=d0)
        outlier = ok & np.isfinite(tail) & (tail < outlier_p)
        d0_vec[outlier] = 0.0            # no shrinkage for hypervariable proteins
    else:
        outlier = np.zeros(var.shape, bool)

    with np.errstate(invalid="ignore", divide="ignore"):
        post = (d0_vec * s02 + df * np.nan_to_num(var)) / (d0_vec + df)
    post = np.where(np.isfinite(var), post, np.nan)
    return post, s02, d0, d0_vec


def moderated_t(beta, se_unscaled, post_var, df_total):
    """se_unscaled = sqrt(L' (X'WX)^-1 L); df_total = residual df + d0."""
    se = se_unscaled * np.sqrt(post_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2.0 * stats.t.sf(np.abs(t), df=df_total)
    return se, t, p


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg, NaN-safe."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    adj = np.empty(m)
    adj[order] = q
    out[ok] = adj
    return out
