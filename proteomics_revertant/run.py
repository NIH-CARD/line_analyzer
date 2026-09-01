"""Analyse datasets read from disk.

    python -m proteomics_revertant.make_data --outdir data      # once
    python -m proteomics_revertant.run --data data --outdir results

Point `--data` at a directory of datasets, a single dataset prefix, or a .h5
bundle. Nothing about the clone panel is hard-coded: the states, allele dosages,
lineages and plexes all come from `samples.tsv`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import qc
from .columns import write_for as write_columns
from .omnibus import narrate, omnibus
from .analysis import analyse
from .datasets import describe, load_dataset


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
    return {"meta": meta, "benchmark": bench}


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

    (outdir / "run_metadata.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
