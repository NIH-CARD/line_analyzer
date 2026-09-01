"""Simulate label-based proteomics intensities for an allelic series.

Variance structure, from the outside in:

    log2 intensity = protein baseline
                   + dose * per-allele effect          (the biology)
                   + dominance deviation               (recessive / dominant)
                   + clone offset                      (drift + off-target)
                   + lineage offset                    (shared by sibling clones)
                   + plex offset                       (TMT batch)
                   + technical noise (abundance dependent)

Missingness is deliberately MNAR: dropout probability is a logistic function of
the true intensity, so low-abundance proteins go missing preferentially. That is
what makes class-wise median imputation dangerous, and what the presence/absence
track in analysis.py exists to handle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _abundance_sd(mu: np.ndarray) -> np.ndarray:
    """Technical SD as a decreasing function of abundance (real MS behaviour)."""
    return 0.12 + 1.10 * np.exp(-(mu - 15.0) / 3.2)


def simulate(design, seed: int = 20260901, null: bool = False):
    """Set null=True to zero out all true biology, keeping every artefact and
    the missingness mechanism intact. Used to check type-I calibration."""
    rng = np.random.default_rng(seed)
    n = C.N_PROTEINS
    proteins = [f"PROT_{i:04d}" for i in range(n)]
    target_ix = proteins.index(C.TARGET_PROTEIN)
    ko_ix = proteins.index(C.KNOCKOUT_PROTEIN)

    # --- protein-level baselines -------------------------------------------
    baseline = rng.normal(23.0, 2.6, n).clip(15.5, 32.0)
    baseline[target_ix] = C.TARGET_BASELINE
    baseline[ko_ix] = C.KNOCKOUT_BASELINE
    tech_sd = _abundance_sd(baseline)

    # --- true biology ------------------------------------------------------
    per_allele = np.zeros(n)
    pool = [i for i in range(n) if i not in (target_ix, ko_ix)]
    responders = rng.choice(pool, C.N_TRUE_RESPONDERS, replace=False)
    # downstream effects, both directions, per-allele log2
    per_allele[responders] = rng.normal(0.0, C.RESPONDER_SD, C.N_TRUE_RESPONDERS)
    per_allele[target_ix] = C.TARGET_PER_ALLELE_LFC          # the one big effect
    per_allele[ko_ix] = C.KNOCKOUT_PER_ALLELE_LFC

    # a third of responders are recessive: little happens at one copy
    dominance = np.zeros(n)
    recessive = rng.choice(responders, size=C.N_TRUE_RESPONDERS // 3, replace=False)
    dominance[recessive] = -0.65 * per_allele[recessive]
    dominance[target_ix] = -0.35 * per_allele[target_ix]     # partly recessive
    dominance[ko_ix] = C.KNOCKOUT_DOMINANCE                  # fully recessive

    if null:
        per_allele[:] = 0.0
        dominance[:] = 0.0

    # --- clone / lineage artefacts -----------------------------------------
    lineages = sorted({c.lineage for c in design.clones})
    lineage_off = {}
    for lg in lineages:
        off = np.zeros(n)
        if lg != "parental":
            hit = rng.random(n) < 0.02                       # shared off-target set
            off[hit] = rng.normal(0.0, 0.45, hit.sum())
        lineage_off[lg] = off

    clone_off = {}
    for cl in design.clones:
        off = rng.normal(0.0, 0.06, n)                       # diffuse culture drift
        private = rng.random(n) < 0.025                      # clone-private hits
        off[private] += rng.normal(0.0, 0.50, private.sum())
        clone_off[cl.name] = off + lineage_off[cl.lineage]

    # one clone carries a chromosome-arm-style coherent block (CNV mimic)
    if design.n_clones >= 5:
        cnv_clone = design.clones[2].name
        block = slice(600, 690)
        clone_off[cnv_clone][block] += 0.42

    # --- assemble libraries ------------------------------------------------
    rows, values = [], []
    lib_ix = 0
    for cl in design.clones:
        for r in range(cl.n_tech):
            plex = lib_ix % design.plexes
            rows.append(
                dict(
                    library=f"{cl.name}_t{r + 1}",
                    clone=cl.name,
                    state=cl.state,
                    dose=cl.dose,
                    lineage=cl.lineage,
                    tech_rep=r + 1,
                    plex=f"plex{plex + 1}",
                )
            )
            lib_ix += 1

    samples = pd.DataFrame(rows)

    plex_off = {p: rng.normal(0.0, 0.10, n) for p in samples["plex"].unique()}
    # a small per-library loading offset, as after IRS normalisation residual
    lib_off = rng.normal(0.0, 0.05, len(samples))

    true_mat = np.zeros((n, len(samples)))
    for j, s in samples.iterrows():
        d = s["dose"]
        het = 1.0 if d == 1 else 0.0
        mu = (
            baseline
            + d * per_allele
            + het * dominance
            + clone_off[s["clone"]]
            + plex_off[s["plex"]]
            + lib_off[j]
        )
        true_mat[:, j] = mu + rng.normal(0.0, tech_sd, n)

    # --- MNAR dropout + a little MCAR --------------------------------------
    # p(missing) rises steeply below ~19 log2 units
    p_mnar = 1.0 / (1.0 + np.exp((true_mat - 18.6) / 0.85))
    p_miss = 1.0 - (1.0 - p_mnar) * (1.0 - 0.015)
    observed = rng.random(true_mat.shape) > p_miss

    obs_mat = np.where(observed, true_mat, np.nan)

    expr = pd.DataFrame(obs_mat, index=proteins, columns=samples["library"].values)
    expr.index.name = "protein_id"

    truth = pd.DataFrame(
        dict(
            protein_id=proteins,
            baseline=baseline,
            true_per_allele_lfc=per_allele,
            true_dominance=dominance,
            is_responder=np.isin(np.arange(n), responders),
            is_target=np.arange(n) == target_ix,
            is_knockout=np.arange(n) == ko_ix,
        )
    ).set_index("protein_id")

    return expr, samples, truth
