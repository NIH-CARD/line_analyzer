"""Load an AnnData (`.h5ad`) object into this pipeline's on-disk dataset format.

The pipeline's native format is three files sharing a prefix (see
`datasets.py`):

    <key>.intensities.tsv   features x libraries, log2 scale, empty = not detected
    <key>.samples.tsv       one row per library: library, clone, state, dose,
                             lineage, tech_rep, plex
    <key>.manifest.json     name, units, provenance

An AnnData object has none of that structure baked in -- it is `obs x var`,
counts live in `.X` or a named layer, and there is no notion of "clone" or
"allele dose" unless you put it there yourself in `.obs`. This module does the
mechanical half of the conversion (orienting the matrix, transforming counts,
assembling `samples.tsv`, validating) and asks you, once, for the four things
it cannot infer: which `.obs` column is the clone, which is the allele dose
(or how to derive it), which is the lineage, and where the counts live.

Library-level use:

    from proteomics_revertant.io_adata import from_anndata
    expr, samples, manifest = from_anndata(
        adata, layer="counts", clone_col="clone", dose_col="dose",
        lineage_col="lineage",
    )
    from proteomics_revertant.datasets import write_dataset
    write_dataset("data", "my_experiment", expr, samples, manifest)

CLI:

    python -m proteomics_revertant.io_adata \\
        --h5ad adata.h5ad --outdir data --key my_experiment \\
        --layer counts --clone-col clone --dose-col genotype \\
        --dose-map '{"wt": 0, "het_edit": 1, "hom_edit": 2, "het_revertant": 1, "hom_revertant": 0}' \\
        --lineage-col lineage --state-col genotype --plex-col batch

--- A caveat worth reading before you run this on RNA-seq ------------------

This format's "empty cell = not detected" convention encodes MNAR dropout,
the missingness pattern of shotgun proteomics. In RNA-seq a zero is an
observed count, not a missing one, and the pipeline's own README says so
directly (`README.md` S9): prefer `filterByExpr` + a negative-binomial fit
for transcriptomics, and "do not impute counts". This converter does not
impute: zeros pass through as `log2(CPM + pseudocount)`, never as blanks,
unless you explicitly pass `--zero-as-missing` to opt into the MNAR
convention (not recommended for count data -- it will treat every silent
gene as a dropout rather than a true zero). Everything downstream of loading
-- the dose-response model, the intersection-union calls, lambda, the
omnibus test -- is agnostic to what generated the numbers, so running
RNA-seq through here is a modelling choice you are making deliberately, not
a limitation of the converter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import validate, write_dataset


def _extract_matrix(adata, layer):
    """Return a dense (genes x libraries) DataFrame: features as rows."""
    if layer in (None, "X"):
        mat = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(
                f"layer {layer!r} not found; available layers: "
                f"{list(adata.layers.keys())}")
        mat = adata.layers[layer]

    if hasattr(mat, "toarray"):  # sparse (scipy.sparse)
        mat = mat.toarray()
    mat = np.asarray(mat)

    genes = adata.var_names.astype(str)
    libraries = adata.obs_names.astype(str)
    # AnnData is obs x var (cells/samples x genes); the pipeline wants
    # features x libraries.
    return pd.DataFrame(mat.T, index=genes, columns=libraries)


def _to_log2_scale(counts: pd.DataFrame, transform: str, pseudocount: float,
                    zero_as_missing: bool) -> pd.DataFrame:
    if (counts.to_numpy() < 0).any():
        raise ValueError(
            "negative values in the chosen layer -- pass raw or normalised "
            "counts, not something already log-transformed or scaled")

    if transform == "log2cpm":
        lib_size = counts.sum(axis=0)
        if (lib_size == 0).any():
            empty = list(lib_size.index[lib_size == 0])
            raise ValueError(f"libraries with zero total counts: {empty}")
        cpm = counts.div(lib_size, axis=1) * 1e6
        log = np.log2(cpm + pseudocount)
    elif transform == "log2":
        log = np.log2(counts + pseudocount)
    elif transform == "none":
        log = counts.copy()
    else:
        raise ValueError(f"unknown transform {transform!r}")

    if zero_as_missing:
        log = log.where(counts > 0)  # zero counts -> blank, MNAR convention

    return log


def _derive_dose(obs: pd.DataFrame, dose_col, state_col, dose_map) -> pd.Series:
    if dose_col is not None and dose_col in obs.columns:
        vals = obs[dose_col]
        if pd.api.types.is_numeric_dtype(vals):
            return vals.astype(int)
        if dose_map is None:
            raise ValueError(
                f"--dose-col {dose_col!r} is non-numeric; pass --dose-map "
                f"to map its values ({sorted(vals.unique())}) to 0/1/2")
        return vals.map(dose_map).astype(int)
    if state_col is not None and dose_map is not None:
        vals = obs[state_col]
        missing = set(vals.unique()) - set(dose_map)
        if missing:
            raise ValueError(f"--dose-map is missing entries for: {sorted(missing)}")
        return vals.map(dose_map).astype(int)
    raise ValueError(
        "could not derive dose -- pass a numeric --dose-col, or a "
        "--state-col together with --dose-map")


def from_anndata(
    adata,
    *,
    layer: str = "counts",
    clone_col: str,
    dose_col: str | None = None,
    state_col: str | None = None,
    lineage_col: str | None = None,
    tech_rep_col: str | None = None,
    plex_col: str | None = None,
    library_col: str | None = None,
    dose_map: dict | None = None,
    transform: str = "log2cpm",
    pseudocount: float = 1.0,
    zero_as_missing: bool = False,
    key: str = "dataset",
    label: str = "",
    source: str = "",
):
    """Convert an in-memory AnnData object to (expr, samples, manifest).

    Only `clone_col` is mandatory. Everything else that cannot be safely
    defaulted raises a specific `ValueError` telling you what to supply --
    this mirrors `datasets.validate`, which runs at the end regardless.
    """
    obs = adata.obs

    if clone_col not in obs.columns:
        raise ValueError(f"clone_col {clone_col!r} not in adata.obs "
                          f"({list(obs.columns)})")

    library = (obs[library_col].astype(str) if library_col else
               adata.obs_names.astype(str).to_series(index=adata.obs_names))
    library = pd.Index(library, name="library")

    dose = _derive_dose(obs, dose_col, state_col, dose_map)

    if state_col is not None and state_col in obs.columns:
        state = obs[state_col].astype(str)
    else:
        state = pd.Series([f"dose_{d}" for d in dose], index=obs.index)

    if lineage_col is not None and lineage_col in obs.columns:
        lineage = obs[lineage_col].astype(str)
    else:
        lineage = obs[clone_col].astype(str)  # each clone its own lineage

    if tech_rep_col is not None and tech_rep_col in obs.columns:
        tech_rep = obs[tech_rep_col]
    else:
        tech_rep = obs.groupby(clone_col, sort=False).cumcount() + 1

    if plex_col is not None and plex_col in obs.columns:
        plex = obs[plex_col]
    else:
        plex = pd.Series(1, index=obs.index)

    samples = pd.DataFrame({
        "library": library.to_numpy(),
        "clone": obs[clone_col].astype(str).to_numpy(),
        "state": state.to_numpy(),
        "dose": dose.to_numpy(),
        "lineage": lineage.to_numpy(),
        "tech_rep": np.asarray(tech_rep),
        "plex": np.asarray(plex),
    })

    counts = _extract_matrix(adata, layer)
    counts.columns = library.to_numpy()  # align to the library ids just built
    expr = _to_log2_scale(counts, transform, pseudocount, zero_as_missing)

    manifest = {
        "key": key,
        "label": label or f"converted from AnnData ({source or 'in-memory'})",
        "units": ("log2 CPM" if transform == "log2cpm" else
                  "log2(x + pseudocount)" if transform == "log2" else "raw"),
        "missing_value": ("empty cell = zero count (MNAR convention, opted in "
                           "via zero_as_missing)" if zero_as_missing else
                           "no missing values -- zero counts are retained as data"),
        "source": {
            "format": "AnnData (.h5ad)",
            "path": source,
            "layer": layer,
            "transform": transform,
            "pseudocount": pseudocount,
            "zero_as_missing": zero_as_missing,
            "n_obs_original": int(adata.n_obs),
            "n_var_original": int(adata.n_vars),
        },
        "notes": (
            "Converted via proteomics_revertant.io_adata. If this is RNA-seq, "
            "see the module docstring / README S9 before trusting the MNAR "
            "missingness track: a zero here is a real observation."
        ),
    }

    validate(expr, samples)
    return expr, samples, manifest


def split_and_convert(
    adata,
    *,
    split_col: str,
    control_value: str = "WT",
    outdir: str,
    key_prefix: str = "",
    **from_anndata_kwargs,
):
    """Split one AnnData into multiple pipeline datasets, one per mutation.

    Rows where `split_col == control_value` (e.g. gene == "WT") are treated
    as shared controls and included in EVERY dataset alongside that
    mutation's own rows -- this is what gives each per-mutation dose axis its
    dose=0 anchor. Every other distinct value of `split_col` becomes its own
    dataset, written as `<outdir>/<key_prefix><value>.{intensities,samples}.tsv`.

    Returns a dict of {mutation_value: prefix_path}.
    """
    obs = adata.obs
    if split_col not in obs.columns:
        raise ValueError(f"split_col {split_col!r} not in adata.obs")

    values = [v for v in obs[split_col].astype(str).unique() if v != control_value]
    if not values:
        raise ValueError(
            f"no non-control values found in {split_col!r} "
            f"(everything equals control_value={control_value!r})")

    written = {}
    for value in sorted(values):
        mask = (obs[split_col].astype(str) == value) | (obs[split_col].astype(str) == control_value)
        sub = adata[mask.to_numpy()].copy()
        key = f"{key_prefix}{value}"
        expr, samples, manifest = from_anndata(sub, key=key, **from_anndata_kwargs)
        prefix = write_dataset(outdir, key, expr, samples, manifest)
        written[value] = prefix
        n_control = int((obs.loc[mask, split_col].astype(str) == control_value).sum())
        print(f"{key}: {mask.sum()} libraries ({n_control} control) "
              f"-> {prefix}.intensities.tsv")
    return written


def _load_h5ad(path):
    try:
        import anndata
    except ImportError as e:
        raise ImportError(
            "reading .h5ad requires the 'anndata' package: "
            "pip install anndata") from e
    return anndata.read_h5ad(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True, help="path to the .h5ad file")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--key", required=True, help="dataset prefix / name")
    ap.add_argument("--label", default="")
    ap.add_argument("--layer", default="counts",
                    help="adata.layers[...] to read; 'X' for adata.X (default: counts)")
    ap.add_argument("--library-col", default=None,
                    help="obs column with unique run/sample ids (default: obs_names)")
    ap.add_argument("--clone-col", required=True)
    ap.add_argument("--dose-col", default=None,
                    help="numeric obs column with allele dose 0/1/2, or a "
                         "categorical one used with --dose-map")
    ap.add_argument("--state-col", default=None,
                    help="obs column with the genotype/state label")
    ap.add_argument("--dose-map", default=None,
                    help='JSON dict mapping state/dose-col values to 0/1/2, '
                         'e.g. \'{"wt": 0, "het": 1, "hom": 2}\'')
    ap.add_argument("--lineage-col", default=None,
                    help="obs column with editing lineage (default: same as clone)")
    ap.add_argument("--tech-rep-col", default=None,
                    help="obs column with replicate index (default: auto-numbered per clone)")
    ap.add_argument("--plex-col", default=None,
                    help="obs column with batch/plex (default: single plex)")
    ap.add_argument("--transform", choices=["log2cpm", "log2", "none"],
                    default="log2cpm")
    ap.add_argument("--pseudocount", type=float, default=1.0)
    ap.add_argument("--zero-as-missing", action="store_true",
                    help="treat zero counts as not-detected (MNAR convention); "
                         "not recommended for RNA-seq, see module docstring")
    ap.add_argument("--hdf5", action="store_true",
                    help="also write a keyed .h5 bundle (requires h5py)")
    ap.add_argument("--split-col", default=None,
                    help="obs column to split into one dataset per value "
                         "(e.g. 'mutation'). When set, --key is used as a "
                         "prefix and one dataset is written per non-control "
                         "value, each including the control rows too.")
    ap.add_argument("--control-value", default="WT",
                    help="value of --split-col treated as shared controls, "
                         "included in every split dataset (default: WT)")
    args = ap.parse_args()

    dose_map = json.loads(args.dose_map) if args.dose_map else None
    adata = _load_h5ad(args.h5ad)

    common_kwargs = dict(
        layer=args.layer,
        clone_col=args.clone_col,
        dose_col=args.dose_col,
        state_col=args.state_col,
        lineage_col=args.lineage_col,
        tech_rep_col=args.tech_rep_col,
        plex_col=args.plex_col,
        library_col=args.library_col,
        dose_map=dose_map,
        transform=args.transform,
        pseudocount=args.pseudocount,
        zero_as_missing=args.zero_as_missing,
        label=args.label,
        source=str(args.h5ad),
    )

    if args.split_col:
        split_and_convert(
            adata,
            split_col=args.split_col,
            control_value=args.control_value,
            outdir=args.outdir,
            key_prefix=f"{args.key}_" if args.key else "",
            **common_kwargs,
        )
        return

    expr, samples, manifest = from_anndata(adata, key=args.key, **common_kwargs)

    from .datasets import describe, design_from_samples
    design = design_from_samples(samples, manifest)
    prefix = write_dataset(args.outdir, args.key, expr, samples, manifest,
                           hdf5=args.hdf5)
    print(describe(expr, samples, design))
    print(f"  -> {prefix}.intensities.tsv / .samples.tsv / .manifest.json"
          f"{' / .h5' if args.hdf5 else ''}")


if __name__ == "__main__":
    main()