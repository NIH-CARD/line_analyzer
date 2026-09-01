"""On-disk dataset format, loader, and design inference.

A dataset is three files sharing a prefix:

    <key>.intensities.tsv   proteins x libraries, log2 intensity, empty = not detected
    <key>.samples.tsv       one row per library, the experimental design
    <key>.manifest.json     dataset name, units, and provenance

Optionally the same content is bundled as a single keyed HDF5 file
(`<key>.h5`) with datasets at `/intensities`, `/protein_id`, `/library`, and
`/samples/<column>`. TSV is canonical; HDF5 is a convenience mirror.

Nothing about the analysis is hard-coded to these particular experiments. The
clone panel, allele dosages, lineages and plexes are all read from
`samples.tsv`, so dropping in your own two files is the whole integration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Clone, Design

SAMPLE_COLUMNS = ["library", "clone", "state", "dose", "lineage", "tech_rep", "plex"]

SAMPLE_COLUMN_DOC = {
    "library": "unique id of one MS run; must match a column of intensities.tsv",
    "clone": "clonal isolate the run came from -- THE UNIT OF REPLICATION",
    "state": "genotype label, e.g. wild_type / het_edit / hom_revertant",
    "dose": "number of mutant alleles carried by the clone: 0, 1 or 2",
    "lineage": "editing event the clone descends from; siblings share off-targets",
    "tech_rep": "replicate index within the clone (separate digest + injection)",
    "plex": "TMT plex / acquisition batch, fitted as a blocking factor",
}


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def write_dataset(outdir, key, expr, samples, manifest, truth=None, hdf5=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / key

    expr = expr.copy()
    expr.index.name = "protein_id"
    expr.round(4).to_csv(f"{prefix}.intensities.tsv", sep="\t", na_rep="")

    samples = samples[SAMPLE_COLUMNS].copy()
    samples.to_csv(f"{prefix}.samples.tsv", sep="\t", index=False)

    manifest = dict(manifest)
    manifest.update(
        n_proteins=int(expr.shape[0]),
        n_libraries=int(expr.shape[1]),
        n_clones=int(samples["clone"].nunique()),
        pct_missing=round(float(100 * expr.isna().to_numpy().mean()), 2),
        column_documentation=SAMPLE_COLUMN_DOC,
        files={
            "intensities": f"{key}.intensities.tsv",
            "samples": f"{key}.samples.tsv",
            "truth": f"{key}.truth.tsv" if truth is not None else None,
        },
    )
    Path(f"{prefix}.manifest.json").write_text(json.dumps(manifest, indent=2))

    if truth is not None:
        truth.round(4).to_csv(f"{prefix}.truth.tsv", sep="\t")

    if hdf5:
        _write_hdf5(f"{prefix}.h5", expr, samples, manifest)
    return prefix


def _write_hdf5(path, expr, samples, manifest):
    import h5py  # optional dependency

    with h5py.File(path, "w") as f:
        f.create_dataset("intensities", data=expr.to_numpy(float),
                         compression="gzip", fillvalue=np.nan)
        f.create_dataset("protein_id", data=np.array(expr.index, dtype="S"))
        f.create_dataset("library", data=np.array(expr.columns, dtype="S"))
        g = f.create_group("samples")
        for col in samples.columns:
            v = samples[col]
            if pd.api.types.is_numeric_dtype(v):
                g.create_dataset(col, data=v.to_numpy())
            else:
                g.create_dataset(col, data=v.astype(str).to_numpy().astype("S"))
        f.attrs["manifest"] = json.dumps(manifest)
        f.attrs["missing_value"] = "NaN means the protein was not detected in that run"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def load_dataset(prefix):
    """Read a dataset by path prefix, directory, or .h5 file.

    Returns (expr, samples, design, truth_or_None).
    """
    prefix = Path(prefix)
    if prefix.suffix == ".h5":
        return _load_hdf5(prefix)
    if prefix.is_dir():
        cands = sorted(prefix.glob("*.intensities.tsv"))
        if len(cands) != 1:
            raise ValueError(
                f"{prefix} holds {len(cands)} datasets; pass the prefix explicitly")
        prefix = Path(str(cands[0]).replace(".intensities.tsv", ""))

    expr = pd.read_csv(f"{prefix}.intensities.tsv", sep="\t", index_col=0)
    samples = pd.read_csv(f"{prefix}.samples.tsv", sep="\t")
    manifest = json.loads(Path(f"{prefix}.manifest.json").read_text())

    truth = None
    tpath = Path(f"{prefix}.truth.tsv")
    if tpath.exists():
        truth = pd.read_csv(tpath, sep="\t", index_col=0)

    validate(expr, samples)
    return expr, samples, design_from_samples(samples, manifest), truth


def _load_hdf5(path):
    import h5py

    with h5py.File(path, "r") as f:
        manifest = json.loads(f.attrs["manifest"])
        proteins = [s.decode() for s in f["protein_id"][:]]
        libraries = [s.decode() for s in f["library"][:]]
        expr = pd.DataFrame(f["intensities"][:], index=proteins, columns=libraries)
        expr.index.name = "protein_id"
        cols = {}
        for col in f["samples"]:
            v = f[f"samples/{col}"][:]
            cols[col] = [x.decode() for x in v] if v.dtype.kind == "S" else v
        samples = pd.DataFrame(cols)[SAMPLE_COLUMNS]
    validate(expr, samples)
    return expr, samples, design_from_samples(samples, manifest), None


def validate(expr, samples):
    """Fail loudly and specifically rather than producing quiet nonsense."""
    missing = [c for c in SAMPLE_COLUMNS if c not in samples.columns]
    if missing:
        raise ValueError(f"samples.tsv is missing required columns: {missing}")
    if not samples["library"].is_unique:
        dup = samples.loc[samples["library"].duplicated(), "library"].tolist()
        raise ValueError(f"duplicate library ids in samples.tsv: {dup}")
    if not expr.index.is_unique:
        raise ValueError("duplicate protein ids in intensities.tsv")

    only_expr = sorted(set(expr.columns) - set(samples["library"]))
    only_samp = sorted(set(samples["library"]) - set(expr.columns))
    if only_expr or only_samp:
        raise ValueError(
            f"library mismatch -- in intensities only: {only_expr}; "
            f"in samples only: {only_samp}")

    bad_dose = sorted(set(samples["dose"]) - {0, 1, 2})
    if bad_dose:
        raise ValueError(f"dose must be 0, 1 or 2; found {bad_dose}")

    per_clone = samples.groupby("clone")[["state", "dose", "lineage"]].nunique()
    inconsistent = per_clone[(per_clone > 1).any(axis=1)]
    if len(inconsistent):
        raise ValueError(
            f"clones with inconsistent state/dose/lineage: {list(inconsistent.index)}")

    arr = expr.to_numpy(float)
    if not np.isfinite(arr[np.isfinite(arr)]).all():
        raise ValueError("intensities contain non-finite values other than blanks")
    if np.nanmax(arr) > 60 or np.nanmin(arr) < 0:
        raise ValueError(
            "intensities look like they are not on a log2 scale "
            f"(range {np.nanmin(arr):.1f} to {np.nanmax(arr):.1f})")
    return True


def design_from_samples(samples, manifest=None) -> Design:
    """Rebuild the clone panel from the sample table. No hard-coded designs."""
    manifest = manifest or {}
    rows = (samples.groupby("clone", sort=False)
            .agg(state=("state", "first"), dose=("dose", "first"),
                 lineage=("lineage", "first"), n_tech=("library", "size"))
            .reset_index()
            .sort_values(["dose", "clone"], kind="stable"))
    clones = [Clone(r["clone"], r["state"], int(r["dose"]), r["lineage"], int(r["n_tech"]))
              for _, r in rows.iterrows()]
    return Design(
        key=manifest.get("key", "dataset"),
        label=manifest.get("label", "loaded from disk"),
        clones=clones,
        notes=manifest.get("notes", ""),
        plexes=int(samples["plex"].nunique()),
    )


def describe(expr, samples, design) -> str:
    """Human-readable summary, printed by the CLI before analysis."""
    lines = [f"{design.label}",
             f"  {expr.shape[0]} proteins x {expr.shape[1]} libraries "
             f"({100 * expr.isna().to_numpy().mean():.1f}% not detected)",
             f"  {design.n_clones} clones, {samples['plex'].nunique()} plex(es), "
             f"residual df = {design.n_clones - 2}",
             "  clone            state                dose  lineage      n_runs  detected"]
    for c in design.clones:
        libs = samples.loc[samples["clone"] == c.name, "library"]
        det = 100 * expr[libs].notna().to_numpy().mean()
        lines.append(f"  {c.name:<16} {c.state:<20} {c.dose:<5} {c.lineage:<12} "
                     f"{c.n_tech:<7} {det:5.1f}%")
    return "\n".join(lines)
