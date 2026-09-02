"""Analyse datasets read from disk.

    python -m proteomics_revertant.make_data --outdir data      # once
    python -m proteomics_revertant.run --data data --outdir results

Point `--data` at a directory of datasets, a single dataset prefix, or a .h5
bundle. Nothing about the clone panel is hard-coded: the states, allele dosages,
lineages and plexes all come from `samples.tsv`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from . import qc
from .columns import write_for as write_columns
from .omnibus import narrate, omnibus
from .analysis import analyse
from .datasets import describe, load_dataset


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
#
# Every number in `results/` should be traceable to the inputs and the software
# that produced it. Recording the interpreter and library versions is not
# ceremony: `ebayes.py` depends on scipy's special functions and the moderated
# variances shift in the last digits between scipy releases, so a result that
# cannot be reproduced exactly is usually a version difference rather than a
# bug. Input files are hashed rather than merely named, because a dataset that
# was regenerated in place is otherwise indistinguishable from one that was not.


def file_digest(path: Path, n_bytes: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed. Returns "" when the file is absent."""
    path = Path(path)
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(n_bytes), b""):
            h.update(block)
    return h.hexdigest()


def input_fingerprint(prefix) -> dict:
    """Hash every file belonging to a dataset prefix."""
    prefix = Path(prefix)
    if prefix.suffix == ".h5":
        return {prefix.name: file_digest(prefix)}
    out = {}
    for suffix in (".intensities.tsv", ".samples.tsv", ".manifest.json",
                   ".truth.tsv"):
        p = Path(str(prefix) + suffix)
        if p.is_file():
            out[p.name] = file_digest(p)
    return out


def environment() -> dict:
    """Interpreter, libraries and machine -- everything needed to reproduce."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def benchmark(results: pd.DataFrame, truth: pd.DataFrame) -> str:
    """Only runs when a dataset ships a .truth.tsv answer key."""
    df = results.set_index("protein_id").join(truth)
    true_pos = (df["is_responder"] | df["is_target"] | df["is_knockout"]).to_numpy()

    def rates(called):
        called = np.asarray(called)
        tp = int((true_pos & called).sum())
        fp = int((~true_pos & called).sum())
        fn = int((true_pos & ~called).sum())
        return tp, fp, fn, fp / max(tp + fp, 1), tp / max(tp + fn, 1)

    primary = (df["call"] != "filtered_no_model").to_numpy() & (
        np.nan_to_num(df["dose_FDR"].to_numpy(), nan=1.0) < 0.05)
    strict = df["call"].isin(["variant_attributable", "presence_absence_hit"]).to_numpy()
    anyhit = df["call"].isin(
        ["variant_attributable", "presence_absence_hit", "dose_trend_only"]).to_numpy()

    fitted = df["dose_logFC"].notna()
    r = np.corrcoef(df.loc[fitted, "dose_logFC"], df.loc[fitted, "true_per_allele_lfc"])[0, 1]
    nulls = df.loc[~true_pos, "dose_P"].dropna()
    t1 = float(np.mean(nulls < 0.05)) if len(nulls) else np.nan

    lines = ["recovery vs the shipped answer key", "-" * 58,
             f"true signal proteins            : {int(true_pos.sum())}",
             f"corr(estimated, true slope)     : {r:.3f}",
             f"type-I rate on null proteins    : {t1:.3f}   (nominal 0.05)", ""]
    for label, mask in (("dose trend, FDR<0.05", primary),
                        ("any call", anyhit),
                        ("strict (IUT + monotone, or PA)", strict)):
        tp, fp, fn, fdr, sens = rates(mask)
        lines.append(f"{label:<32}: TP={tp:<4d} FP={fp:<4d} FN={fn:<4d} "
                     f"empFDR={fdr:.3f} sens={sens:.3f}")

    lines.append("")
    planted = df.index[df["is_target"] | df["is_knockout"]]
    for name in planted:
        t = df.loc[name]
        lines += [f"{name}",
                  f"   true per-allele logFC   : {t['true_per_allele_lfc']:.3f}",
                  f"   estimated dose logFC    : {t['dose_logFC']}  (SE {t['dose_SE']})",
                  f"   dose FDR                : {t['dose_FDR']}",
                  f"   track / call            : {t['track']} / {t['call']}",
                  f"   pct missing             : {t['pct_missing']}",
                  f"   presence-absence P/FDR  : {t['pa_trend_P']} / {t['pa_trend_FDR']}",
                  ""]
    return "\n".join(lines)


def find_datasets(path: Path):
    path = Path(path)
    if path.suffix == ".h5" or path.name.endswith(".intensities.tsv"):
        return [path]
    if path.is_dir():
        found = sorted(Path(str(p).replace(".intensities.tsv", ""))
                       for p in path.glob("*.intensities.tsv"))
        if found:
            return found
    return [path]


def run_one(prefix, outdir: Path) -> dict:
    t_start = time.perf_counter()
    expr, samples, design, truth = load_dataset(prefix)
    print(describe(expr, samples, design))
    results, meta = analyse(expr, samples, design)

    key = design.key
    results.to_csv(outdir / f"{key}_results.tsv", sep="\t", index=False)
    # column reference, derived from the same design object the columns were
    write_columns(design, outdir, results)

    # the top-level answer: did the edit change the proteome at all?
    overall = omnibus(expr, samples, design, results, analyse_fn=analyse)
    story = narrate(overall)
    (outdir / f"{key}_overall.txt").write_text(story + "\n")
    flat = {k: v for k, v in overall.items()
            if not isinstance(v, (list, dict))}
    pd.DataFrame([flat]).to_csv(outdir / f"{key}_overall.tsv", sep="\t", index=False)
    print()
    print(story)

    qc.missingness_report(expr, samples).to_csv(
        outdir / f"{key}_qc_missingness.tsv", sep="\t", index=False)
    qc.missingness_vs_abundance(expr).to_csv(
        outdir / f"{key}_qc_mnar_profile.tsv", sep="\t", index=False)
    qc.artefact_scale(results, design).to_csv(
        outdir / f"{key}_qc_artefact_scale.tsv", sep="\t", index=False)
    qc.null_calibration(results, design).to_csv(
        outdir / f"{key}_qc_null_calibration.tsv", sep="\t", index=False)
    qc.directional_balance(results).to_csv(
        outdir / f"{key}_qc_directional_balance.tsv", sep="\t", index=False)
    qc.coherent_block_scan(expr, samples).to_csv(
        outdir / f"{key}_qc_coherent_blocks.tsv", sep="\t")
    qc.clone_correlation(expr, samples).to_csv(
        outdir / f"{key}_qc_clone_correlation.tsv", sep="\t")

    bench = None
    if truth is not None:
        bench = benchmark(results, truth)
        (outdir / f"{key}_benchmark.txt").write_text(bench + "\n")

    meta["overall"] = {k: v for k, v in overall.items()
                       if not isinstance(v, (list, dict))}
    meta["rows_written"] = int(results.shape[0])
    meta["columns_written"] = int(results.shape[1])
    meta["calls"] = results["call"].value_counts().to_dict()
    meta["tracks"] = results["track"].value_counts().to_dict()

    # provenance for this dataset
    meta["inputs"] = input_fingerprint(prefix)
    meta["outputs"] = sorted(p.name for p in outdir.glob(f"{key}_*"))
    meta["elapsed_seconds"] = round(time.perf_counter() - t_start, 3)

    # the calibration constants, promoted out of `overall` so that every run
    # reports them in one predictable place. lambda_hat is the estimate,
    # lambda_applied is what the analysis actually used -- they differ whenever
    # the estimate falls below 1, because deflating a standard error is never
    # the conservative choice.
    meta["calibration"] = {
        "lambda_hat": overall.get("lambda_hat"),
        "lambda_lo": overall.get("lambda_lo"),
        "lambda_hi": overall.get("lambda_hi"),
        "lambda_applied": overall.get("lambda_applied"),
        "n_artefact_contrasts": overall.get("n_artefact_contrasts"),
        "residual_df": overall.get("residual_df"),
        "proteins_tested": overall.get("proteins_tested"),
    }
    return {"meta": meta, "benchmark": bench}


def calibration_table(summary: dict) -> pd.DataFrame:
    """One row per dataset: the calibration constants, side by side.

    Lambda is per-run and not comparable to a fixed threshold, so the useful
    view is across datasets: a panel whose lambda sits well above its
    neighbours' is one whose standard errors are least trustworthy.
    """
    rows = []
    for key, meta in summary.items():
        cal = meta.get("calibration", {})
        rows.append({"dataset": key,
                     "n_clones": meta.get("n_clones"),
                     **{k: cal.get(k) for k in
                        ("residual_df", "proteins_tested",
                         "n_artefact_contrasts", "lambda_hat", "lambda_lo",
                         "lambda_hi", "lambda_applied")}})
    return pd.DataFrame(rows)


def write_run_log(outdir: Path, summary: dict, started: str, elapsed: float,
                  argv: list, data_arg: str) -> Path:
    """Human-readable provenance record for the whole invocation."""
    env = environment()
    L = ["Run log -- proteomics_revertant.run",
         "=" * 70,
         f"started (UTC)      : {started}",
         f"elapsed (s)        : {elapsed:.1f}",
         f"command            : {' '.join(argv)}",
         f"input path         : {data_arg}",
         f"output directory   : {outdir.resolve()}",
         "",
         "Environment",
         "-" * 70]
    L += [f"  {k:<16} : {v}" for k, v in env.items()]
    L += ["",
          "Datasets",
          "-" * 70]
    for key, meta in summary.items():
        cal = meta.get("calibration", {})
        L += [f"  {key}",
              f"    clones           : {meta.get('n_clones')}",
              f"    residual df      : {cal.get('residual_df')}",
              f"    features tested  : {cal.get('proteins_tested')}",
              f"    lambda_hat       : {cal.get('lambda_hat')} "
              f"[{cal.get('lambda_lo')}, {cal.get('lambda_hi')}]",
              f"    lambda_applied   : {cal.get('lambda_applied')}",
              f"    rows x columns   : {meta.get('rows_written')} x "
              f"{meta.get('columns_written')}",
              f"    elapsed (s)      : {meta.get('elapsed_seconds')}",
              "    inputs (sha256)  :"]
        for name, digest in (meta.get("inputs") or {}).items():
            L.append(f"       {name:<38} {digest[:16]}")
        L.append("")
    text = "\n".join(L)
    path = outdir / "run_log.txt"
    path.write_text(text + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data",
                    help="directory of datasets, a dataset prefix, or a .h5 file")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = find_datasets(args.data)
    if not datasets:
        raise SystemExit(f"no datasets found under {args.data}")

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()

    summary = {}
    for prefix in datasets:
        print("=" * 70)
        res = run_one(prefix, outdir)
        print()
        print(json.dumps(res["meta"], indent=2, default=str))
        if res["benchmark"]:
            print()
            print(res["benchmark"])
        summary[res["meta"]["design"]] = res["meta"]

    elapsed = time.perf_counter() - t0

    # machine-readable provenance, and the same facts as prose
    (outdir / "run_metadata.json").write_text(json.dumps(
        {"started_utc": started,
         "elapsed_seconds": round(elapsed, 3),
         "command": [sys.executable, "-m", "proteomics_revertant.run",
                     "--data", str(args.data), "--outdir", str(args.outdir)],
         "environment": environment(),
         "datasets": summary},
        indent=2, default=str))

    cal = calibration_table(summary)
    cal.to_csv(outdir / "calibration_summary.tsv", sep="\t", index=False)
    log = write_run_log(outdir, summary, started, elapsed,
                        [sys.executable, "-m", "proteomics_revertant.run",
                         "--data", str(args.data), "--outdir", str(args.outdir)],
                        str(args.data))

    print()
    print("standard-error calibration across datasets")
    print(cal.to_string(index=False))
    print(f"\nwrote outputs to {outdir.resolve()}")
    print(f"provenance: {log.name}, run_metadata.json, calibration_summary.tsv")


if __name__ == "__main__":
    main()
