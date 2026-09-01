"""Generate the two example datasets on disk. Run once; the analysis then reads
files, never this module.

    python -m proteomics_revertant.make_data --outdir data [--hdf5]

The data are synthetic so that a tester can check the pipeline against a known
answer -- each dataset ships a `.truth.tsv` answer key listing the planted
per-allele effects. Replace these three files with your own and nothing in the
analysis changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config as C
from .datasets import describe, write_dataset
from .simulate import simulate

MANIFESTS = {
    "A_homozygous_wt": dict(
        key="A_homozygous_wt",
        label="Background A: homozygous wild-type parental, full allelic series",
        units="log2 normalised reporter intensity (TMT, internal reference scaled)",
        missing_value="empty cell = protein not detected in that run (never imputed)",
        notes=(
            "Five clones spanning 0, 1, 2, 1, 0 mutant alleles. Doses 0 and 1 are "
            "each carried by two clones with different editing histories, so the "
            "residual of the dose model is clone-to-clone artefact rather than "
            "pipetting noise."),
        planted_truth=(
            "PROT_0042: abundant, -2.2 log2 per allele, quantified in every run. "
            "PROT_0777: recessive on/off, absent from the homozygous edit only. "
            "40 further responders at N(0, 0.45) per allele, a third recessive."),
    ),
    "A_replicated_x3": dict(
        key="A_replicated_x3",
        label="Background A with 3 independently derived clones per genotype state",
        units="log2 normalised reporter intensity (TMT, internal reference scaled)",
        missing_value="empty cell = protein not detected in that run (never imputed)",
        notes=(
            "15 clones, 30 runs, 13 residual degrees of freedom. Each of the five "
            "genotype states is replicated by clones from independent editing "
            "rounds, so an off-target site hit in one round cannot imitate a dose "
            "response. Technical replication is 2 runs per clone: with a fixed "
            "budget, clones buy inference and injections do not."),
        planted_truth=(
            "Same planted effects as background A: PROT_0042 at -2.2 log2 per "
            "allele, PROT_0777 recessive on/off, 40 further responders."),
    ),
    "A_replicated_x5": dict(
        key="A_replicated_x5",
        label="Background A with 5 independently derived clones per genotype state",
        units="log2 normalised reporter intensity (TMT, internal reference scaled)",
        missing_value="empty cell = protein not detected in that run (never imputed)",
        notes=(
            "25 clones, 50 runs, 23 residual degrees of freedom. The fully "
            "replicated version: lambda and the editing-noise floor are estimated "
            "from twenty artefact contrasts, and the permutation null is large "
            "enough that the global p-value is not floor-limited."),
        planted_truth=(
            "Same planted effects as background A."),
    ),
    "B_heterozygous_wt_minimal": dict(
        key="B_heterozygous_wt_minimal",
        label="Background B, minimal: one clone per genotype state (underpowered)",
        units="log2 normalised reporter intensity (TMT, internal reference scaled)",
        missing_value="empty cell = protein not detected in that run (never imputed)",
        notes=(
            "Three clones, one residual degree of freedom, no matched-dose pair. "
            "Shipped deliberately as the negative control for the design: it "
            "still recovers the large planted effect but loses the on/off "
            "protein and every corroborating contrast."),
        planted_truth=(
            "Same planted effects as the other backgrounds; the difference is "
            "how few of them survive with three clones."),
    ),
    "B_heterozygous_wt": dict(
        key="B_heterozygous_wt",
        label="Background B: heterozygous carrier parental, two clones per edited state",
        units="log2 normalised reporter intensity (TMT, internal reference scaled)",
        missing_value="empty cell = protein not detected in that run (never imputed)",
        notes=(
            "The parental line already carries one mutant allele, so only three "
            "genotype states exist. Two independently derived isolates of each "
            "edited state restore replication at matched dose."),
        planted_truth="Same planted effects as background A.",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--hdf5", action="store_true",
                    help="also write a keyed .h5 bundle (requires h5py)")
    args = ap.parse_args()

    out = Path(args.outdir)
    for key, manifest in MANIFESTS.items():
        design = C.DESIGNS[key]
        expr, samples, truth = simulate(design, seed=args.seed)
        manifest = dict(manifest, seed=args.seed)
        prefix = write_dataset(out, key, expr, samples, manifest,
                               truth=truth, hdf5=args.hdf5)
        print(describe(expr, samples, design))
        print(f"  -> {prefix}.intensities.tsv / .samples.tsv / .manifest.json"
              f"{' / .h5' if args.hdf5 else ''}\n")


if __name__ == "__main__":
    main()
