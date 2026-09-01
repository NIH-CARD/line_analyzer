"""Experimental designs for the two editing backgrounds.

Dose = number of mutant alleles carried by the clone. The whole analysis hangs
off this axis: clonal drift and off-target artefacts have no reason to track
allele dosage, so a monotone dose response across independently derived clones
is much harder to fake than any single pairwise contrast.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Clone:
    name: str
    state: str          # biological state label
    dose: int           # mutant allele count: 0, 1 or 2
    lineage: str        # editing history; clones sharing a lineage share drift
    n_tech: int = 4     # technical replicates (separate digests + injections)


@dataclass
class Design:
    key: str
    label: str
    clones: List[Clone]
    # named linear contrasts over model coefficients, filled in by model.py
    notes: str = ""
    plexes: int = 2     # TMT plexes; clones are split across them on purpose

    @property
    def doses(self) -> List[int]:
        return [c.dose for c in self.clones]

    @property
    def n_clones(self) -> int:
        return len(self.clones)

    def state_of(self) -> Dict[str, str]:
        return {c.name: c.state for c in self.clones}


# ---------------------------------------------------------------------------
# Background A: parental line is homozygous wild type (dose 0)
#
# Full allelic series. Doses 0 and 1 are each represented by TWO clones with
# DIFFERENT editing histories (WT vs hom-revertant at dose 0; het edit vs het
# revertant at dose 1). Those matched-dose pairs are what give the residual its
# meaning -- the residual variance is clone-to-clone artefact, not pipetting.
# 5 clones, 2 parameters (intercept + dose) => 3 residual df.
# ---------------------------------------------------------------------------
BACKGROUND_A = Design(
    key="A_homozygous_wt",
    label="Homozygous WT background (full allelic series)",
    clones=[
        Clone("WT",      "wild_type",         dose=0, lineage="parental"),
        Clone("HetEd",   "het_edit",          dose=1, lineage="edit_r1"),
        Clone("HomEd",   "hom_edit",          dose=2, lineage="edit_r1"),
        Clone("HetRev",  "het_revertant",     dose=1, lineage="edit_r2"),
        Clone("HomRev",  "hom_revertant",     dose=0, lineage="edit_r2"),
    ],
    notes=(
        "Matched-dose pairs: (WT, HomRev) at dose 0 and (HetEd, HetRev) at dose 1. "
        "Differences within those pairs are pure editing/clonal artefact."
    ),
)

# ---------------------------------------------------------------------------
# Background B: parental line is already a heterozygous carrier (dose 1)
#
# Only three genotype states exist. With one clone per state that is 3 clones
# and 1 residual df, which is not enough to say anything about a genotype. The
# fix is independent clones per state, so the default carries two independent
# isolates of each edited state -- 5 clones, 3 residual df, and within-state
# clone pairs that estimate the artefact directly.
# Pass --minimal to run.py to see the 3-clone version and how thin it is.
# ---------------------------------------------------------------------------
BACKGROUND_B = Design(
    key="B_heterozygous_wt",
    label="Heterozygous carrier background (2 independent clones per edited state)",
    clones=[
        Clone("Par",     "parental_het",      dose=1, lineage="parental"),
        Clone("HomEdA",  "hom_edit",          dose=2, lineage="edit_r1a"),
        Clone("HomEdB",  "hom_edit",          dose=2, lineage="edit_r1b"),
        Clone("HomRevA", "hom_revertant",     dose=0, lineage="edit_r2a"),
        Clone("HomRevB", "hom_revertant",     dose=0, lineage="edit_r2b"),
    ],
    notes=(
        "Within-state clone pairs (HomEdA/HomEdB, HomRevA/HomRevB) estimate the "
        "editing artefact; the parental het is the only dose-1 clone."
    ),
)

BACKGROUND_B_MINIMAL = Design(
    key="B_heterozygous_wt_minimal",
    label="Heterozygous carrier background (one clone per state -- underpowered)",
    clones=[
        Clone("Par",     "parental_het",      dose=1, lineage="parental"),
        Clone("HomEd",   "hom_edit",          dose=2, lineage="edit_r1"),
        Clone("HomRev",  "hom_revertant",     dose=0, lineage="edit_r2"),
    ],
    notes="1 residual df. Dose is fully confounded with clone identity.",
    plexes=1,
)

# ---------------------------------------------------------------------------
# Replicated panels: the same five genotype states, but with several
# INDEPENDENTLY DERIVED clones of each.
#
# This is the design the earlier panels keep pointing at. Two things break with
# one clone per state and both are fixed by replication rather than by more mass
# spectrometry:
#
#   1. the permutation null has only 30 usable relabellings, so the global
#      p-value cannot go below about 0.07 however strong the biology is;
#   2. lambda and the editing-noise floor rest on one or two artefact contrasts,
#      so they are estimated with almost no precision.
#
# Technical replication drops to 2 runs per clone here on purpose. Given a fixed
# budget, clones buy inference and injections do not: once clone variance is
# non-zero the standard error of a genotype contrast is floored by tau^2/n_clones
# regardless of how many times each clone is run.
# ---------------------------------------------------------------------------

_STATES = [("wild_type", 0), ("het_edit", 1), ("hom_edit", 2),
           ("het_revertant", 1), ("hom_revertant", 0)]


def replicated_design(n_rep: int, n_tech: int = 2) -> Design:
    """Five genotype states, `n_rep` independently derived clones of each.

    Lineages follow how the clones would really be made: within replicate r the
    het and hom edits come out of one editing round (`ed_r`) and the two
    revertants out of a second round on that background (`rev_r`). Replicates
    do not share lineages with each other, which is the whole point -- an
    off-target site hit in one editing round cannot masquerade as a dose
    response reproduced across independent rounds.
    """
    clones = []
    for r in range(1, n_rep + 1):
        for state, dose in _STATES:
            if state == "wild_type":
                lineage = f"parental_{r}"
            elif "revertant" in state:
                lineage = f"rev_{r}"
            else:
                lineage = f"ed_{r}"
            short = {"wild_type": "WT", "het_edit": "HetEd", "hom_edit": "HomEd",
                     "het_revertant": "HetRev", "hom_revertant": "HomRev"}[state]
            clones.append(Clone(f"{short}{r}", state, dose, lineage, n_tech))
    return Design(
        key=f"A_replicated_x{n_rep}",
        label=f"Homozygous WT background, {n_rep} independent clones per state",
        clones=clones,
        notes=(f"{n_rep * 5} clones, {n_rep * 5 * n_tech} runs. Every genotype "
               f"state is replicated by independently derived clones, so the "
               f"editing-noise floor and lambda are estimated from "
               f"{n_rep * 5 - 5 + (n_rep - 1)} artefact contrasts rather than "
               f"one or two, and the permutation null is large enough that the "
               f"global p-value is not floor-limited."),
        plexes=max(2, (n_rep * 5 * n_tech) // 10),
    )


BACKGROUND_A_REP3 = replicated_design(3)
BACKGROUND_A_REP5 = replicated_design(5)

DESIGNS = {d.key: d for d in (
    BACKGROUND_A, BACKGROUND_B, BACKGROUND_B_MINIMAL,
    BACKGROUND_A_REP3, BACKGROUND_A_REP5)}

# --- simulation constants --------------------------------------------------

N_PROTEINS = 1000
TARGET_PROTEIN = "PROT_0042"        # the one big effect: abundant, dose-dependent
TARGET_BASELINE = 27.0              # high enough to stay quantifiable at dose 2
TARGET_PER_ALLELE_LFC = -2.2        # hom edit sits 4.4 log2 below WT

KNOCKOUT_PROTEIN = "PROT_0777"      # recessive on/off protein, exercises the PA track
KNOCKOUT_BASELINE = 22.0
KNOCKOUT_PER_ALLELE_LFC = -2.9      # dose 2 falls into the dropout zone
KNOCKOUT_DOMINANCE = 2.7            # recessive: one copy looks like wild type

N_TRUE_RESPONDERS = 40              # proteins with a real dose response
RESPONDER_SD = 0.45                 # per-allele log2 effect SD among responders
MIN_OBS_PER_STATE = 2               # presence filter (counts, never percentages)
