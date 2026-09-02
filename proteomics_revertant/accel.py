"""Accelerated kernels: closed-form weighted fit and the resident design sweep.

Two ideas do all the work here, and only the second of them needs a GPU.

**Hoist the invariants.** The permutation test relabels which clone carries
which allele dosage. Clone means, observation counts and within-clone variances
do not depend on dosage at all -- only the c x p design matrix changes. The
reference implementation recomputes everything from the intensity matrix for
every relabelling. `sweep_designs` computes the clone summaries once and drives
P designs against them. That is a pure algorithmic win and it applies on CPU.

**Keep the data resident.** Once the summaries are on the device, a sweep of P
designs touches them P times with no host round trip. This is the only part of
the pipeline with enough parallel work to interest a GPU, and it is exactly the
part that dominates the runtime.

The p = 2 case (intercept + dose) gets a closed form: a symmetric 2x2 normal
matrix has an explicit inverse, so there is no batched `solve` and no LU at all.
Measured on CPU that is 2.7-4.9x faster than the einsum-and-solve version, and
it removes the one kernel that GPUs handle poorly at tiny matrix sizes.

Precision. The kernel is memory-bound, so float32 halves the traffic and very
nearly halves the time. It is safe for the sweep statistic, which only needs to
rank permutations, and is NOT the default for reported coefficients: the
residual sum of squares is a difference of similar-magnitude quantities and
loses roughly seven significant digits in float32 on a 5-point fit.
`tests.py` measures that difference rather than assuming it.
"""

from __future__ import annotations

import numpy as np

from .backend import Backend, plan_chunks, resolve


# ---------------------------------------------------------------------------
# weighted fit
# ---------------------------------------------------------------------------


def fit_closed_form(xp, means, W, X):
    """Weighted least squares for p = 2, via the explicit 2x2 inverse.

    Returns (beta0, beta1, s2, fitted). No `solve`, no per-protein work.
    """
    x0, x1 = X[:, 0], X[:, 1]
    s00 = W @ (x0 * x0)
    s01 = W @ (x0 * x1)
    s11 = W @ (x1 * x1)
    Wm = W * means
    t0 = Wm @ x0
    t1 = Wm @ x1
    det = s00 * s11 - s01 * s01
    b0 = (s11 * t0 - s01 * t1) / det
    b1 = (s00 * t1 - s01 * t0) / det
    fitted = b0[:, None] * x0[None, :] + b1[:, None] * x1[None, :]
    resid = means - fitted
    dfree = X.shape[0] - 2
    s2 = xp.einsum("nc,nc->n", resid * W, resid) / dfree
    return b0, b1, s2, fitted


def fit_general(xp, means, W, X):
    """Weighted least squares for arbitrary p. Batched normal equations."""
    Xw = X[None, :, :] * W[:, :, None]
    XtWX = xp.einsum("ncp,cq->npq", Xw, X)
    XtWy = xp.einsum("ncp,nc->np", Xw, means)
    beta = xp.linalg.solve(XtWX, XtWy[..., None])[..., 0]
    fitted = beta @ X.T
    resid = means - fitted
    dfree = X.shape[0] - X.shape[1]
    s2 = xp.einsum("nc,nc->n", resid * W, resid) / dfree
    return beta, s2, fitted, XtWX


def weighted_fit(xp, means, W, X):
    """Dispatch to the closed form when p = 2, otherwise the general path."""
    if X.shape[1] == 2:
        b0, b1, s2, fitted = fit_closed_form(xp, means, W, X)
        beta = xp.stack([b0, b1], axis=1) if hasattr(xp, "stack") else \
            np.stack([b0, b1], axis=1)
        return beta, s2
    beta, s2, _, _ = fit_general(xp, means, W, X)
    return beta, s2


# ---------------------------------------------------------------------------
# the resident sweep
# ---------------------------------------------------------------------------


def tau2_statistic(xp, beta_dose, se_dose, lam=1.0):
    """mean(beta^2) - lambda * mean(SE^2), the omnibus dispersion statistic."""
    return (xp.einsum("n,n->", beta_dose, beta_dose) / beta_dose.shape[0]
            - lam * xp.einsum("n,n->", se_dose, se_dose) / se_dose.shape[0])


# Below this many clone-summary elements (proteins x clones) a GPU loses to
# numpy: the kernel is memory-bound, so a small sweep is over before the PCIe
# round trip and the CUDA context have paid for themselves. Measured on an
# RTX 5080 against numpy, 30 designs, float64 -- CPU wins at 125k elements
# (5k x 25), the GPU wins from 250k up, and the two are level near 100k:
#
#     elements     CPU ms    GPU ms
#        25,000       4.5      24.6
#       125,000      19.8      37.6
#       250,000      98.7      46.4
#     2,500,000    1447.2     278.8
#
# This only steers `backend="auto"`. An explicit backend is always honoured --
# `bench.py` has to be able to ask for the slow one on purpose.
GPU_MIN_ELEMENTS = 250_000


def sweep_designs(means, W, designs, unscaled_var=None, lam=1.0,
                  backend="auto", device="cuda", dtype=np.float64,
                  chunk=None, return_beta=False, return_raw=False):
    """Run P designs against one resident copy of the clone summaries.

    `means` and `W` are (proteins x clones), already reduced from libraries;
    `designs` is a list of (clones x p) matrices. `unscaled_var[k]` is the
    sqrt(L' (X'WX)^-1 L) factor for the dose coefficient under design k, which
    the caller can precompute since it does not depend on the data -- but it is
    computed here when omitted.

    NOTE `unscaled_var` is only correct when the weights are the same for every
    protein. With ragged missingness (X'WX)^-1 varies per protein, so leave it
    None and let the per-protein factor be computed here.

    With `backend="auto"` the device is chosen by problem size, not merely by
    whether a GPU exists: see GPU_MIN_ELEMENTS. A panel of a few thousand
    analytes runs faster on the CPU, and picking the GPU there made the hoisted
    permutation path *slower* than the reference it is supposed to beat.

    Returns the tau2 statistic per design, and optionally the dose coefficients.
    """
    if isinstance(backend, Backend):
        b = backend
    elif backend == "auto":
        b = (resolve("auto", device) if means.size >= GPU_MIN_ELEMENTS
             else resolve("numpy"))
    else:
        b = resolve(backend, device)
    xp = b.xp
    n, c = means.shape
    p = designs[0].shape[1]

    if chunk is None:
        chunk = plan_chunks(n, c, p, b, dtype=dtype, n_designs=len(designs))

    d_designs = [b.asarray(X, dtype=dtype) for X in designs]

    taus = np.zeros(len(designs))
    sq_beta = np.zeros(len(designs))
    sq_se = np.zeros(len(designs))
    betas = [[] for _ in designs] if (return_beta or return_raw) else None
    s2s = [[] for _ in designs] if return_raw else None
    uvars = [[] for _ in designs] if return_raw else None

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        m = b.asarray(means[start:stop], dtype=dtype)
        w = b.asarray(W[start:stop], dtype=dtype)

        for k, X in enumerate(d_designs):
            beta, s2 = weighted_fit(xp, m, w, X)
            dose = beta[:, 1]
            if unscaled_var is not None:
                u = float(unscaled_var[k])
                se2 = s2 * (u ** 2)
            else:
                # (X'WX)^-1 [1,1] for the dose coefficient, per protein
                x0, x1 = X[:, 0], X[:, 1]
                s00 = w @ (x0 * x0)
                s01 = w @ (x0 * x1)
                s11 = w @ (x1 * x1)
                se2 = s2 * (s00 / (s00 * s11 - s01 * s01))
            # accumulate sums, not means: chunks may differ in size
            sq_beta[k] += float(b.to_numpy(xp.einsum("n,n->", dose, dose)))
            sq_se[k] += float(b.to_numpy(xp.einsum("n->", se2)))
            if return_beta or return_raw:
                betas[k].append(b.to_numpy(dose))
            if return_raw:
                s2s[k].append(b.to_numpy(s2))
                uvars[k].append(b.to_numpy(se2 / s2))
        b.sync()

    taus = sq_beta / n - lam * sq_se / n
    if return_raw:
        # beta, raw residual variance, and the (X'WX)^-1[1,1] factor per design,
        # so the caller can apply empirical-Bayes moderation exactly as
        # analysis.analyse does before forming the statistic
        return (taus,
                [np.concatenate(x) for x in betas],
                [np.concatenate(x) for x in s2s],
                [np.concatenate(x) for x in uvars])
    if return_beta:
        return taus, [np.concatenate(x) for x in betas]
    return taus


def describe(backend="auto", device="cuda") -> str:
    b = resolve(backend, device)
    free = b.free_bytes()
    line = f"{b.label}"
    if free:
        line += f", {free / 1e9:.1f} GB free on device"
    return line
