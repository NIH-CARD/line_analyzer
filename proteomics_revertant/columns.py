"""Data dictionary for `<key>_results.tsv`, generated from the design.

The contrast columns are named after the clone panel, so a hard-coded list would
go stale the moment someone edits `samples.tsv`. This module therefore derives
the dictionary the same way `analysis.py` derives the columns -- from
`contrasts_for(design, ...)` and the clone states themselves -- so the two
cannot drift apart. `tests.py` asserts the two sets match exactly.

    python -m proteomics_revertant.columns --data data --outdir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analysis import build_design, contrasts_for, iut_pair

# ---------------------------------------------------------------------------
# columns that exist regardless of the clone panel
# ---------------------------------------------------------------------------

FIXED = [
    # (column, group, dtype, description, computed_from, how_to_read)
    ("protein_id", "identity", "str",
     "Protein or protein-group identifier, taken verbatim from the first column "
     "of the intensity file.",
     "input file",
     "Join key. One row per protein, always 1000 rows for the shipped datasets, "
     "including proteins that could not be tested."),

    ("track", "coverage", "str",
     "Which analysis route the protein took: `continuous` (quantified widely "
     "enough to model), `presence_absence` (missing from an entire genotype "
     "state, so tested on detection counts), `presence_absence+continuous` "
     "(both), or `insufficient` (fewer than 2 observations in every state).",
     "observation counts per state vs MIN_OBS_PER_STATE",
     "Read this before any p-value. A `presence_absence` protein has no "
     "meaningful `dose_logFC`; its evidence lives in the `pa_*` columns."),

    ("n_libraries", "coverage", "int",
     "Total number of MS runs in the dataset. Constant down the column.",
     "width of the intensity matrix",
     "Denominator for `pct_missing`. Not the sample size for inference -- that "
     "is the clone count."),

    ("n_observed", "coverage", "int",
     "Number of libraries in which this protein was quantified.",
     "count of finite cells in the protein's row",
     "Low values weaken everything downstream. Compare against `obs_*` to see "
     "whether the loss is spread evenly or concentrated in one genotype."),

    ("pct_missing", "coverage", "float (%)",
     "Percentage of libraries with no quantification for this protein.",
     "100 * (1 - n_observed / n_libraries)",
     "Reported for transparency, never used as a filter. A protein knocked out "
     "by the edit is *supposed* to be missing in one third of runs."),

    ("mean_log2_intensity", "coverage", "float (log2)",
     "Mean log2 intensity across the libraries where the protein was observed.",
     "nanmean of the protein's row",
     "Abundance proxy. Missingness and technical variance both rise steeply as "
     "this falls, so treat low-abundance hits with more caution."),

    ("sigma2_tech", "variance", "float (log2^2)",
     "Pooled within-clone (technical) variance: replicate injections and "
     "digests of the same clone.",
     "pooled sum of squares about each clone mean / sum(n_c - 1)",
     "Compare with `clone_var_moderated`. If technical variance dominates, more "
     "injections help; if clone variance dominates, only more clones help."),

    ("clone_var_moderated", "variance", "float (log2^2)",
     "The empirical-Bayes moderated clone-level residual variance -- the error "
     "term every contrast is tested against.",
     "squeeze_var(): (d0*s0^2 + df*s^2) / (d0 + df)",
     "This is clone-to-clone noise, not pipetting noise. Its square root is "
     "roughly the log2 scale of the editing/drift artefact for this protein."),

    ("df_residual", "variance", "float",
     "Residual degrees of freedom of the additive dose fit for this protein: "
     "clones with data minus model parameters.",
     "n_clones_with_data - ncol(design)",
     "Usually 3 for a 5-clone panel. If you ever see a number near the library "
     "count, the model has been pseudoreplicated."),

    ("df_moderated", "variance", "float",
     "Degrees of freedom actually used for the t-tests: `df_residual` plus the "
     "prior degrees of freedom borrowed across proteins.",
     "df_residual + d0 (per protein; d0 = 0 for variance outliers)",
     "The gap between this and `df_residual` is how much the 1000-protein prior "
     "is doing for you. `d0` is reported in run_metadata.json."),

    ("variance_outlier", "variance", "bool",
     "True if the protein is genuinely hypervariable across clones and was "
     "exempted from variance shrinkage.",
     "upper-tail F test of the dose-SATURATED residual against the fitted prior",
     "These proteins keep their own (large) variance rather than borrowing the "
     "bulk prior. Deliberately judged on the saturated residual so that a "
     "recessive protein is not penalised for lack of fit to a linear model."),

    ("dominance_estimable", "variance", "bool",
     "True if the dose-saturated model (intercept + dose + heterozygote "
     "deviation) was estimable for this protein.",
     "rank and residual df of the secondary design after dropping empty clones",
     "When False, `dominance_*` is blank and the hypervariability test falls "
     "back to the additive residual."),

    ("lambda_artifact", "calibration", "float",
     "Genomic-control inflation factor (lambda_GC) estimated from the artefact "
     "contrasts. REPORTED, NOT APPLIED. Constant down the column.",
     "median over artefact contrasts of "
     "median(chi2_1^-1(1 - p)) / median(chi2_1), unfloored",
     "A diagnostic, not a correction. The artefact contrasts carry no variant "
     "signal by construction, so this audits the standard errors: 1.0 means the "
     "median test statistic is the expected size, above 1.1 means nominal SEs "
     "are too small. No recalibrated p-value column is emitted -- whether to "
     "apply genomic control is the analyst's decision. Values below 1 are "
     "normal on these panels and must NOT be divided out; see the README."),

    ("sign_concordant", "integration", "bool",
     "True if the two corroborating contrasts move in the same direction.",
     "sign(edit_vs_rev_logFC) == sign(edit_vs_baseline_logFC)",
     "A protein that goes up versus the revertant and down versus baseline is "
     "not showing a coherent variant effect; the IUT p-value is set to 1."),

    ("iut_P", "integration", "float",
     "Intersection-union p-value for 'differs from the revertant AND from "
     "baseline, in the same direction'.",
     "max of two TWO-SIDED p-values, or 1 if the directions disagree",
     "The maximum of two p-values is itself a valid p-value for the "
     "intersection null, so no extra multiplicity correction is needed here."),

    ("iut_FDR", "integration", "float",
     "Benjamini-Hochberg adjustment of `iut_P` across proteins.",
     "BH over iut_P",
     "The strict evidence standard. Conservative by construction -- a protein "
     "must clear both contrasts, not just the more favourable one."),

    ("monotone_in_dose", "integration", "bool",
     "True if the mean abundance is monotone across the three allele dosages.",
     "sign consistency of successive differences of dose-group means",
     "A required condition for a `variant_attributable` call. Clone artefacts "
     "have no reason to respect dose order, so this is cheap corroboration."),

    ("pa_trend_z", "detection", "float",
     "Cochran-Armitage trend statistic for detection against allele dosage.",
     "sum(dose_i * (detected_i - n_i * p_bar)) / sqrt(V)",
     "Negative means the protein is detected less often as mutant allele count "
     "rises. Reported for direction; the p-value is exact, not from this z."),

    ("pa_trend_P", "detection", "float",
     "Exact permutation p-value for the detection trend, TWO-SIDED: a protein "
     "detected more often at higher allele counts scores the same as one "
     "detected less often.",
     "enumeration of the hypergeometric null, counting |T| >= |T_observed|",
     "The evidence for on/off proteins. Blank for proteins quantified in every "
     "state -- there is no detection pattern to test."),

    ("pa_trend_FDR", "detection", "float",
     "Benjamini-Hochberg adjustment of `pa_trend_P`, over the detection track "
     "only.",
     "BH over pa_trend_P",
     "A separate testing family from the continuous track, which is why the two "
     "FDR columns are not comparable to each other."),

    ("pa_hit", "verdict", "bool",
     "True if `pa_trend_FDR` < 0.05.",
     "pa_trend_FDR < 0.05",
     "Convenience flag; the same threshold that drives `call`."),

    ("artifact_flag", "verdict", "bool",
     "True if any artefact contrast is significant at FDR < 0.05 for this "
     "protein.",
     "any(artifact_*_FDR < 0.05)",
     "A caveat, not a filter. It says the clones disagree at matched dosage, so "
     "read any call for this protein with suspicion -- but absence of "
     "significance is not evidence the reversion was clean (use an equivalence "
     "test for that)."),

    ("call", "verdict", "str",
     "Single-word verdict, in priority order: `variant_attributable` (IUT "
     "FDR<0.05 and monotone), `presence_absence_hit` (detection trend "
     "FDR<0.05), `dose_trend_only` (dose FDR<0.05 but fails the corroborating "
     "pair), `not_significant`, `filtered_no_model` (too sparse to test).",
     "priority cascade at the end of analyse()",
     "`dose_trend_only` is not a failure -- it is the primary estimand without "
     "corroboration, common for real but modest effects in a 5-clone panel."),
]

_ROLE_BLURB = {
    "primary": ("PRIMARY ESTIMAND. Per-allele change: the slope of log2 "
                "abundance on mutant allele count."),
    "secondary": ("Deviation of the heterozygote from the additive line. Large "
                  "positive values (relative to the dose slope's sign) indicate "
                  "a recessive response."),
    "corroborating": ("Corroborating pairwise contrast, one of the two combined "
                      "by the intersection-union test."),
    "artefact": ("ARTEFACT CONTROL. Two clones of the SAME allele dosage but "
                 "different editing histories, so this estimates clonal drift "
                 "and off-target noise with no variant signal in it."),
}


def _contrast_entries(con, design):
    role = _ROLE_BLURB[con.role]
    if con.kind == "clones":
        members = ", ".join(f"{w:+g}*{c}" for c, w in con.spec.items())
        source = f"linear combination of clone means: {members}"
        unit = "log2 ratio"
    else:
        source = f"`{con.spec}` coefficient of the weighted clone-level fit"
        unit = "log2 per allele" if con.spec == "dose" else "log2"

    return [
        (f"{con.name}_logFC", f"contrast: {con.name}", f"float ({unit})",
         f"{role} Effect estimate for `{con.name}`.",
         source,
         "Effect size on the log2 scale; 1.0 is a two-fold change. Compare "
         "against the artefact contrasts' robust SD before calling it real."),
        (f"{con.name}_SE", f"contrast: {con.name}", "float",
         f"Moderated standard error of `{con.name}_logFC`.",
         "sqrt(clone_var_moderated * sum(c_i^2 / w_i)) over clone means",
         "Built from the clone-level error term, so it reflects clone-to-clone "
         "variability rather than injection reproducibility."),
        (f"{con.name}_t", f"contrast: {con.name}", "float",
         f"Moderated t-statistic for `{con.name}`.",
         f"{con.name}_logFC / {con.name}_SE",
         "Tested on `df_moderated` degrees of freedom, not `df_residual`."),
        (f"{con.name}_P", f"contrast: {con.name}", "float",
         f"TWO-SIDED p-value for `{con.name}`. Tests against zero in either "
         f"direction; an increase and a decrease of the same size get the same "
         f"p-value.",
         "2 * t.sf(|t|, df_moderated)  -- the factor of 2 is the second tail",
         "Raw p-value. Mildly anticonservative in the tails because clone "
         "artefacts are heavy-tailed; the FDR columns are the ones to act on."),
    ]


def data_dictionary(design) -> pd.DataFrame:
    """Every column of `<key>_results.tsv` for this design, in file order."""
    names_sec = build_design(design, secondary=True)[1]
    cons = contrasts_for(design, names_sec)
    fixed = {c[0]: c for c in FIXED}
    rows = []

    def add(entry):
        rows.append(dict(zip(
            ["column", "group", "dtype", "description", "computed_from",
             "how_to_read"], entry)))

    # order mirrors analysis.py exactly
    add(fixed["protein_id"])
    add(fixed["track"])
    for c in ("n_libraries", "n_observed", "pct_missing"):
        add(fixed[c])

    doses = {c.state: c.dose for c in design.clones}
    states = sorted({c.state for c in design.clones},
                    key=lambda s: (doses[s], s))   # ties broken by name, see analysis.py
    for st in states:
        clones = [c.name for c in design.clones if c.state == st]
        add((f"obs_{st}", "coverage", "int",
             f"Number of libraries in which the protein was quantified across "
             f"the `{st}` clone(s) ({', '.join(clones)}, {doses[st]} mutant "
             f"allele(s)).",
             "count of finite cells in those libraries",
             "A zero here with non-zero counts elsewhere is the on/off pattern "
             "that routes the protein to the detection track."))

    for c in ("mean_log2_intensity", "sigma2_tech", "clone_var_moderated",
              "df_residual", "df_moderated", "variance_outlier",
              "dominance_estimable"):
        add(fixed[c])

    fdr_roles = {"primary", "secondary", "artefact"}
    for con in cons:
        for e in _contrast_entries(con, design):
            add(e)
        if con.role in fdr_roles:
            add((f"{con.name}_FDR", f"contrast: {con.name}", "float",
                 f"Benjamini-Hochberg adjustment of `{con.name}_P` across "
                 f"proteins.",
                 f"BH over {con.name}_P",
                 "For the dose contrast this is the headline result. For "
                 "artefact contrasts it flags proteins where clones of matched "
                 "dosage genuinely disagree."))

    # analysis.py emits the equivalence columns in a second pass, after every
    # contrast block, so the dictionary follows the same order
    for con in cons:
        if con.role != "artefact":
            continue
        add((f"{con.name}_equiv_P", f"contrast: {con.name}", "float",
             f"Equivalence (TOST) p-value for `{con.name}`: positive evidence "
             f"that the two clones agree to within the negligibility bound, "
             f"rather than merely failing to differ.",
             "max of two ONE-sided t tests against +/- equivalence_delta_log2 "
             "(default 0.2 log2, about a 15% change); Schuirmann 1987",
             "Small means the clones are demonstrably equivalent -- the "
             "reversion is clean at this protein. A large ordinary `_P` only "
             "means the experiment lacked power to see a difference, which is "
             "a much weaker claim."))
        add((f"{con.name}_equiv_FDR", f"contrast: {con.name}", "float",
             f"Benjamini-Hochberg adjustment of `{con.name}_equiv_P`.",
             f"BH over {con.name}_equiv_P",
             "Proteins passing this are the positive evidence that editing left "
             "no residue. Quote this rather than a non-significant artefact "
             "p-value."))

    add(fixed["lambda_artifact"])

    c1, c2 = iut_pair(design)
    have_iut = {con.name for con in cons} >= {c1, c2}
    if have_iut:
        for c in ("sign_concordant", "iut_P", "iut_FDR"):
            add(fixed[c])
    add(fixed["monotone_in_dose"])
    for c in ("pa_trend_z", "pa_trend_P", "pa_trend_FDR"):
        add(fixed[c])
    add(fixed["pa_hit"])
    if any(con.role == "artefact" for con in cons):
        add(fixed["artifact_flag"])
    add(fixed["call"])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------

_READING_ORDER = """\
## How to read a row

Work through the columns in this order; stopping early is usually correct.

1. **`track`** -- can this protein be tested at all, and by which test?
   `insufficient` means stop. `presence_absence` means ignore every `logFC` and
   go straight to `pa_trend_P`.
2. **`call`** -- the one-word verdict. `variant_attributable` is the strict
   standard: significant against both the revertant and baseline, same
   direction, monotone in dose.
3. **`dose_logFC` and `dose_FDR`** -- the primary estimand, a per-allele log2
   change. This is the number to quote.
4. **`artifact_*_logFC`** -- how far apart clones of *identical* genotype sit.
   If your effect is not comfortably larger than these, it is inside the
   editing-noise floor and should not be called.
5. **`clone_var_moderated` and `df_residual`** -- the error term and how much
   information stands behind it. `df_residual` should be small (clone count
   minus parameters). If it looks like the library count, something is wrong.

Two thresholds are baked into `call`: FDR < 0.05 on the intersection-union test
plus monotonicity for `variant_attributable`, and FDR < 0.05 on the detection
trend for `presence_absence_hit`. Everything else in the file is the evidence
behind those, exposed so you can set your own bar.

## What the file does not contain

No imputed values. A blank cell in the source intensities stays blank, and a
protein missing from an entire genotype is tested on detection counts rather
than filled in. No percentage-missing filter either -- that would delete the
knockout you are looking for.
"""


def _worked_example(design, df: pd.DataFrame, results: pd.DataFrame) -> str:
    """Annotate one real row so the columns are read against actual numbers."""
    if results is None or results.empty:
        return ""
    r = results.copy()
    rank = r["iut_P"] if "iut_P" in r else r["dose_P"]
    if rank.notna().any():
        row = r.loc[rank.idxmin()]
    else:
        row = r.iloc[0]

    lead = [
        "## Worked example",
        "",
        f"The strongest hit in this dataset, `{row['protein_id']}`, read column "
        f"by column.",
        "",
        "| column | value | what it says |",
        "|---|---|---|",
    ]

    def fmt(v):
        if isinstance(v, float):
            if pd.isna(v):
                return "(blank)"
            return f"{v:.4g}"
        return str(v)

    picks = [
        ("track", "quantified widely enough for the continuous model"),
        ("n_observed", "runs with a quantification, out of `n_libraries`"),
        ("dose_logFC", "per-allele log2 change -- the number to quote"),
        ("dose_SE", "moderated standard error, built on clone-level noise"),
        ("df_residual", "degrees of freedom behind that error term"),
        ("df_moderated", "after borrowing the prior from the other proteins"),
        ("dose_FDR", "primary result after Benjamini-Hochberg"),
        ("monotone_in_dose", "abundance moves in one direction across 0, 1, 2 alleles"),
        ("sign_concordant", "both corroborating contrasts agree in direction"),
        ("iut_P", "max of the two corroborating p-values"),
        ("iut_FDR", "strict standard: significant against revertant AND baseline"),
        ("variance_outlier", "not judged hypervariable, so it kept the shared prior"),
        ("lambda_artifact", "artefact-contrast calibration for the whole dataset"),
        ("call", "the verdict"),
    ]
    body = []
    for col, note in picks:
        if col in row.index:
            body.append(f"| `{col}` | {fmt(row[col])} | {note} |")

    art = [c for c in results.columns
           if c.startswith("artifact_") and c.endswith("_logFC")]
    if art:
        vals = ", ".join(f"{fmt(row[c])}" for c in art)
        body.append(f"| `artifact_*_logFC` | {vals} | clones of matched dosage "
                    f"differ by this much -- the noise floor to beat |")

    tail = [
        "",
        "The reading: the effect is far larger than the matched-dosage clone "
        "differences on the same row, it is monotone in allele count, and it "
        "survives being tested against both the revertant and the baseline "
        "rather than only the more favourable comparison.",
        "",
    ]
    return "\n".join(lead + body + tail)


def to_markdown(design, df: pd.DataFrame, results: pd.DataFrame = None) -> str:
    clones = ", ".join(f"{c.name} ({c.state}, dose {c.dose}, lineage "
                       f"{c.lineage})" for c in design.clones)
    head = [
        f"# Column reference -- `{design.key}_results.tsv`",
        "",
        f"**{design.label}**",
        "",
        f"Clone panel: {clones}.",
        "",
        f"{design.n_clones} clones, {design.n_clones - 2} residual degrees of "
        f"freedom for the additive dose model. One row per protein.",
        "",
        design.notes,
        "",
        _READING_ORDER,
        "",
    ]
    example = _worked_example(design, df, results)
    if example:
        head += [example, ""]
    head += ["## Every column", ""]
    body = []
    for group, sub in df.groupby("group", sort=False):
        body.append(f"### {group}")
        body.append("")
        body.append("| column | type | meaning | computed from | how to read it |")
        body.append("|---|---|---|---|---|")
        for _, r in sub.iterrows():
            cells = [f"`{r['column']}`", r["dtype"], r["description"],
                     r["computed_from"], r["how_to_read"]]
            body.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        body.append("")
    return "\n".join(head + body)


def write_for(design, outdir: Path, results: pd.DataFrame = None):
    df = data_dictionary(design)
    tsv = outdir / f"{design.key}_columns.tsv"
    md = outdir / f"{design.key}_columns.md"
    df.to_csv(tsv, sep="\t", index=False)
    md.write_text(to_markdown(design, df, results))
    return df, tsv, md


def main():
    from .datasets import load_dataset
    from .run import find_datasets

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for prefix in find_datasets(args.data):
        _, _, design, _ = load_dataset(prefix)
        df, tsv, md = write_for(design, outdir)
        print(f"{design.key}: {len(df)} columns documented -> {md.name}, {tsv.name}")


if __name__ == "__main__":
    main()
