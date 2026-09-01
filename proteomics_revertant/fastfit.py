"""Batched version of the per-protein fit in `analysis.py`.

The loop in `analyse()` solves a separate weighted least squares per protein, on
a 5x2 design. Almost all of the runtime is Python overhead and ~2 tiny SVDs per
protein, not arithmetic. But the design matrix and the weights depend only on
the protein's per-clone observation counts, and there are very few distinct
count vectors in a real dataset. Grouping on that vector lets one factorisation
serve thousands of proteins and turns the whole thing into a handful of matrix
multiplies.

This is the same arithmetic, not an approximation: within a group every protein
sees the identical `pinv`, so the results agree to floating-point noise.
`tests.py` asserts agreement against the loop on the shipped data.
"""

from __future__ import annotations

import numpy as np


def fit_batched(means, counts, W, X_pri, X_sec, fit_mask, cons,
                clone_names, names_pri, names_sec):
    n_prot = means.shape[0]
    est = {c.name: np.full(n_prot, np.nan) for c in cons}
    unscaled = {c.name: np.full(n_prot, np.nan) for c in cons}
    s2 = np.full(n_prot, np.nan)
    df_resid = np.full(n_prot, np.nan)
    s2_sec = np.full(n_prot, np.nan)
    df_sec = np.full(n_prot, np.nan)
    used_secondary = np.zeros(n_prot, bool)
    clone_ix = {nm: i for i, nm in enumerate(clone_names)}

    rows = np.flatnonzero(fit_mask)
    if rows.size == 0:
        return est, unscaled, s2, df_resid, s2_sec, df_sec, used_secondary

    # group proteins that share an observation-count vector: same design, same
    # weights, therefore one factorisation for the whole group
    keys = counts[rows]
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    inverse = inverse.ravel()

    for gi in range(first.size):
        idx = rows[inverse == gi]
        cnt = counts[idx[0]]
        ok = cnt > 0
        k = int(ok.sum())
        w = W[idx[0], ok]
        Y = means[np.ix_(idx, np.flatnonzero(ok))]        # (m, k)

        Xg = X_pri[ok]
        p = Xg.shape[1]
        if np.linalg.matrix_rank(Xg) < p or k - p < 1:
            continue
        Wm = np.diag(w)
        A = np.linalg.pinv(Xg.T @ Wm @ Xg) @ Xg.T @ Wm     # (p, k)

        A_sec = None
        if "dominance" in names_sec:
            Xs = X_sec[ok]
            if np.linalg.matrix_rank(Xs) == Xs.shape[1] and k - Xs.shape[1] >= 1:
                A_sec = np.linalg.pinv(Xs.T @ Wm @ Xs) @ Xs.T @ Wm

        resid = Y - (Y @ A.T) @ Xg.T
        dfree = k - p
        s2[idx] = np.einsum("ij,j,ij->i", resid, w, resid) / dfree
        df_resid[idx] = dfree
        used_secondary[idx] = A_sec is not None

        if A_sec is not None:
            Xs = X_sec[ok]
            d_s = k - Xs.shape[1]
            if d_s >= 1:
                r_s = Y - (Y @ A_sec.T) @ Xs.T
                s2_sec[idx] = np.einsum("ij,j,ij->i", r_s, w, r_s) / d_s
                df_sec[idx] = d_s

        pos = {c: j for j, c in enumerate(np.flatnonzero(ok))}
        for con in cons:
            if con.kind == "coef":
                if con.spec in names_pri:
                    a = A[names_pri.index(con.spec)]
                elif A_sec is not None and con.spec in names_sec:
                    a = A_sec[names_sec.index(con.spec)]
                else:
                    continue
            else:
                a = np.zeros(k)
                bad = False
                for cname, wt in con.spec.items():
                    j = pos.get(clone_ix[cname])
                    if j is None:
                        bad = True
                        break
                    a[j] = wt
                if bad:
                    continue
            est[con.name][idx] = Y @ a
            unscaled[con.name][idx] = float(np.sqrt(np.sum(a**2 / w)))

    return est, unscaled, s2, df_resid, s2_sec, df_sec, used_secondary
