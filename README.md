# Allelic-series analysis for edited / revertant clone panels

Differential abundance analysis for experiments in which a variant is introduced
into a cell line and then reverted, with proteomics or transcriptomics readout.
The pipeline estimates the per-allele effect of the variant, calibrates it
against the experiment's own editing noise, and reports both per-feature results
and a single genome-wide answer.

Pure Python (`numpy`, `scipy`, `pandas`). No R, no compiled extensions. The
`limma` functions it depends on are reimplemented in `ebayes.py` and validated
against their published definitions.

```bash
pip install -r requirements.txt
python -m proteomics_revertant.make_data --outdir data --hdf5   # synthetic examples
python -m proteomics_revertant.run  --data data --outdir results
python -m proteomics_revertant.pca  --data data --outdir results
python -m proteomics_revertant.tests                            # 37 checks
```

Point `--data` at your own directory of datasets to replace the examples. The
clone panel, allele dosages, lineages and plexes are read from `samples.tsv`;
nothing about the experimental design is hard-coded.

Throughout this document *feature* means a row of the intensity matrix — a
protein group in proteomics, a transcript or gene in transcriptomics. The method
is indifferent to which.

---

## 1. The problem

Comparing an edited clone to a parental clone does not measure the variant. Two
clones grown side by side diverge through culture adaptation, copy-number
change, and off-target editing, and all of that appears as differential
abundance. With one clone per genotype the variant effect and the clone effect
are perfectly confounded.

Reverting the edit breaks the confound. A revertant derived from the edited clone
carries the same accumulated drift and the same off-target load, but not the
causal allele. Anything shared by the edited clone and its revertant is not the
variant.

The pipeline turns this into an estimand by placing every clone on one axis —
the number of mutant alleles it carries — and fitting the slope along it:

```
   WT        het edit    hom edit    het revertant   hom revertant
   0 alleles  1 allele    2 alleles   1 allele        0 alleles
   └──────────┴───────────┴───────────┴───────────────┘
                 dose response fitted through these
```

Two clones sit at dosage 0 and two at dosage 1, with different editing
histories. The fitted line cannot pass through all five points, and the residual
scatter is exactly the disagreement between clones that should be identical.
**Every standard error in the output is computed against that residual.** The
question each feature is asked is therefore not "does the edited clone differ"
but "is the dose response larger than the disagreement between clones of the same
genotype". Editing noise is the measuring stick, not a correction applied
afterwards.

Allele dosage is the axis rather than a series of pairwise comparisons because
clonal drift has no reason to align with allele count. A feature that steps
cleanly from 0 to 1 to 2 alleles, and steps back when the allele is reverted, is
difficult for an artefact to imitate.

## 2. Method

**Unit of replication.** Technical replicates are collapsed to clone means;
residual degrees of freedom are (clones − parameters), never (runs − parameters).
Clone means are weighted by `1/(ρ + (1−ρ)/n_c)` with a genome-wide consensus
intraclass correlation, the closed form of the compound-symmetry weighting in
`limma::duplicateCorrelation`. This is the single most consequential choice in
the pipeline: treating runs as replicates would inflate the degrees of freedom
from 3 to 17 in a five-clone panel and make every p-value wrong by orders of
magnitude.

**Variance moderation.** Three to twenty-three residual degrees of freedom per
feature is too thin alone, so residual variances are shrunk toward a scaled-F
prior fitted across all features (Smyth 2004). Genuinely hypervariable features
are exempted from shrinkage; the hypervariability test uses the residual from the
dose-saturated model, so that a recessive feature is not penalised for lack of
fit to a linear model.

**No imputation.** Filtering is on presence in counts — at least two observations
in at least one genotype state — never on percent missing, which would delete the
on/off features a knockout produces. Features absent from an entire genotype are
routed to an exact permutation test on detection counts rather than filled in and
t-tested.

**Two-sided throughout.** Every reported per-feature p-value tests against zero in
either direction. The only one-sided quantities are the equivalence (TOST)
columns and the omnibus permutation p-values, whose statistics are non-negative
dispersion measures, as for an F-test. Both are one-sided by construction rather
than by choice.

**Intersection–union calls.** A feature is called variant-attributable only if it
differs from both the revertant and the baseline, in the same direction, and is
monotone in dosage. `max(p₁, p₂)` is a valid p-value for that intersection, so
Benjamini–Hochberg is applied once across features with no further correction.

**Calibration (λ).** The matched-dosage clone pairs contain no variant signal by
construction, so their p-values are a *designed* null and audit the standard
errors directly. λ is the genomic-control inflation factor, computed exactly as
in a GWAS — see §2.1. It is reported with a bootstrap interval, and it is
**reported only**: the pipeline does not perform genomic control, because
whether to correct is a decision for the analyst rather than the pipeline.

**The omnibus test.** Whether the edit perturbed the system *at all* is a separate
question from which features moved, and it is answered by reassigning which clone
carries which allele dosage and refitting. Clones stay intact; only the
clone-to-dosage link is broken. §4.2 covers the resulting columns.

### 2.1 λ, the genomic-control inflation factor

λ answers one question: **are the error bars honest?**

It is the same statistic used in genome-wide association studies, computed the
same way, from p-values:

```
λ = median( χ²₁ ⁻¹(1 − p) ) / median( χ²₁ )        median(χ²₁) = 0.4549
```

Each p-value is converted to the chi-square statistic on one degree of freedom
that would have produced it, and the median of those is compared against the
median expected under a well-behaved null. Read it as a ratio of observed to
expected test-statistic magnitude:

| λ | Interpretation |
|---|---|
| ≈ 1.0 | The median test statistic is the expected size. See the caveat below before reading this as "well calibrated". |
| 1.1 – 1.2 | Mild inflation. p-values optimistic by roughly √λ. Consider correcting downstream. |
| > 1.2 | Substantial unmodelled structure. Investigate before trusting any p-value. |
| < 1.0 | The bulk of the null is narrower than the reference. Expected in these panels — see below. Never used to deflate standard errors. |

**λ below 1 is normal here, and is not corrected.** On true-null simulations λ
settles in the 0.77–0.88 band and stays there however many clones are added; it
does not converge to 1. The reason is that clone artefacts are heavy-tailed
rather than normal, which narrows the bulk of the null distribution (λ < 1) while
*widening* its extreme tail — at 100 clones the rate of p < 0.001 reaches roughly
4.6× nominal. These are the same phenomenon at different quantiles, and they move
in opposite directions, so no single scale factor fixes both. Genomic control
divides by λ, so applying a λ of 0.87 would inflate every statistic and make the
deep tail worse — which is the concrete reason this pipeline reports λ and stops
there. If you do choose to apply it downstream, guard the direction yourself
(`max(λ, 1)`) rather than dividing by whatever came out.

**A λ near 1 on a small panel is not reassurance.** At five clones λ is about
1.0, but the rate of p < 0.05 on the designed null is around 0.007 against a
nominal 0.05 — the entire distribution is compressed by variance moderation
borrowing heavily against three residual degrees of freedom. λ near 1 there is a
coincidence of the median, not evidence that the error bars are right. Read it
next to `n_artefact_contrasts` and the interval width.

Two design choices are worth stating explicitly.

*It is computed on the artefact contrasts, not on the dose contrast.* In a GWAS
the null is the assumption that most markers do nothing. Here the null is a
designed feature of the panel: clones of identical allele dosage genuinely have
no variant signal between them. That is a stronger footing than the GWAS
assumption. Computing λ on `dose_P` instead would fold real variant signal into
the calibration constant and shrink genuine effects.

*It is derived from p-values rather than from t-statistics.* Under the null a
p-value is uniform whatever the degrees of freedom, so the reference median is
the same constant for a 3-df panel and a 300-df one, and λ stays comparable
across designs of different size. Taking the median rather than the mean is what
makes it robust to the minority of features carrying real signal.

**The interval matters as much as the point estimate.** With five clones there
are only two artefact contrasts and the bootstrap interval is wide. An interval
spanning 1.0 means the data cannot distinguish honest standard errors from
modestly inflated ones — a statement about the panel's size, not a clean bill of
health. A panel with no two clones sharing a dosage cannot measure λ at all, and
the pipeline says so rather than defaulting silently to 1.

λ appears in four places, and they agree by construction:

| Location | Content |
|---|---|
| `<key>_qc_null_calibration.tsv` | One row per artefact contrast — the most granular view. |
| `<key>_overall.tsv` | `lambda_hat` = median across those contrasts, with a bootstrap CI. |
| `<key>_results.tsv` | `lambda_artifact` = the same estimate, repeated per row for convenience. |
| `calibration_summary.tsv` | Every dataset in the run, side by side. |

All four report the same unfloored estimate, so a value below 1 stays visible.
`lambda_applied` is `max(λ, 1)` and is **not** a p-value correction — nothing in
`results.tsv` is scaled by it. It is used only as a conservative multiplier on
the estimation-error term when the omnibus test subtracts noise from the
genome-wide effect size, and it is a no-op wherever λ ≤ 1.

## 3. Output files

One set per dataset, plus three run-level files.

| File | Question it answers |
|---|---|
| `<key>_overall.txt` | Did the edit change the system at all, and by how much? Plain prose. |
| `<key>_overall.tsv` | The same answer as 42 machine-readable fields. |
| `<key>_results.tsv` | Which individual features changed. One row per feature. |
| `<key>_columns.md` / `.tsv` | Generated data dictionary: every column of *this* dataset's results file, with the formula behind it and a worked example. |
| `<key>_qc_*.tsv` | Seven diagnostics — missingness, MNAR profile, noise scale, calibration, directional balance, coherent blocks, clone correlation. |
| `<key>_pca_*.tsv` / `.png` / `.txt` | Genotype-coloured PCA: scores, loadings, ranked influence, figure, reading. |
| `<key>_benchmark.txt` | Recovery against the answer key, for datasets that ship one. |
| `run_log.txt` | Human-readable provenance for the whole invocation. |
| `run_metadata.json` | The same provenance, machine-readable. |
| `calibration_summary.tsv` | λ and design size for every dataset in the run. |

Because the results columns depend on the clone panel — a five-clone panel emits
62, a twenty-five-clone panel 209 — the per-dataset dictionary in
`<key>_columns.md` is generated from the design and is always authoritative for
your data. §4.1 below explains the *families* those columns fall into, which is
what stays constant.

## 4. Reading the results

### 4.1 `<key>_results.tsv` — one row per feature

**Read it in this order. Stopping early is usually correct.**

1. `track` — can this feature be tested at all, and by which route?
2. `call` — the one-word verdict.
3. `dose_logFC` and `dose_FDR` — the primary estimand and its significance.
4. `artifact_*_logFC` — the noise floor. Is your effect comfortably larger?
5. `clone_var_moderated` and `df_residual` — the error term and its weight.

#### Identity and coverage

| Column | Meaning |
|---|---|
| `protein_id` | Feature identifier, verbatim from the first column of the intensity file. |
| `track` | Analysis route taken. `continuous` = quantified widely enough to model. `presence_absence` = missing from an entire genotype, so tested on detection counts instead. `insufficient` = too sparse to test; stop reading the row. |
| `n_libraries` | Total runs in the dataset. Constant down the column. |
| `n_observed` | Runs in which this feature was quantified. |
| `pct_missing` | Percentage of runs with no quantification. **Never used as a filter** — see §2. |
| `obs_<state>` | Runs with a quantification within each genotype state. One column per state present in the panel. |
| `mean_log2_intensity` | Mean log2 intensity over the runs where the feature was observed. |

#### Variance and degrees of freedom

| Column | Meaning |
|---|---|
| `sigma2_tech` | Pooled within-clone variance — replicate injections and digests of the same clone. Technical noise only. |
| `clone_var_moderated` | The empirical-Bayes moderated clone-level residual variance. **This is the error term every contrast is tested against.** |
| `df_residual` | Residual degrees of freedom of the dose fit: clones with data minus model parameters. For a five-clone panel this is 3. **If it ever resembles the run count, the analysis has been corrupted by pseudoreplication.** |
| `df_moderated` | Degrees of freedom actually used for the t-tests — `df_residual` plus the prior degrees of freedom borrowed across features. Typically 9–12 for a five-clone panel. |
| `variance_outlier` | `True` if the feature is genuinely hypervariable across clones and was exempted from shrinkage. |
| `dominance_estimable` | `True` if the dose-saturated model (intercept + dose + heterozygote deviation) could be fitted. |

#### Contrasts

Every contrast emits the same five columns. Learn the pattern once:

| Suffix | Meaning |
|---|---|
| `_logFC` | The effect estimate, in log2 units. 1.0 = a doubling. |
| `_SE` | Moderated standard error of that estimate. |
| `_t` | `logFC / SE`, the moderated t-statistic. |
| `_P` | Two-sided p-value. An increase and a decrease of the same size score identically. |
| `_FDR` | Benjamini–Hochberg adjustment of `_P` across features. |

The contrasts themselves:

| Contrast | Role | What it asks |
|---|---|---|
| `dose` | **Primary** | Per-allele change — the slope of log2 abundance on mutant allele count. **This is the number to quote.** |
| `dominance` | Secondary | Deviation of the heterozygote from the additive line. Large values relative to the dose slope indicate a recessive or dominant response rather than an additive one. Only present when estimable. |
| `edit_vs_rev` | Corroborating | Edited clones versus their revertants. |
| `edit_vs_baseline` | Corroborating | Edited clones versus the unedited baseline. |
| `artifact_<A>_vs_<B>` | **Artefact control** | Two clones of the *same* allele dosage but different editing history. Contains no variant signal by construction — this is the experiment's own measurement of clonal drift and off-target noise. One per matched-dosage pair. |
| `revertant_vs_baseline` | Artefact | Aggregate revertant-versus-baseline. Present only when replication makes it distinct from a single clone pair. |

Two additional columns appear on artefact contrasts:

| Column | Meaning |
|---|---|
| `artifact_*_equiv_P` | Equivalence (TOST) p-value: **positive** evidence that two clones agree to within the negligibility bound (default 0.2 log2), rather than mere failure to detect a difference. One-sided by construction. |
| `artifact_*_equiv_FDR` | Benjamini–Hochberg adjustment of the above. |

> **Using the artefact contrasts.** These are the reason the design exists. An
> effect that is not comfortably larger than the artefact `logFC` values sits
> inside the editing-noise floor of your panel and should not be called,
> whatever its p-value.

#### Calibration

| Column | Meaning |
|---|---|
| `lambda_artifact` | The genomic-control factor for this dataset (see §2.1), unfloored. Constant down the column. **Reported as a diagnostic; never applied.** |

No corrected p-value column is emitted. `dose_P` and `dose_FDR` are the
uncorrected two-sided results, and the test suite asserts that they fit an
uncorrected reconstruction better than any λ-corrected one. If you want genomic
control, it is one step downstream and under your control:

```python
from scipy import stats
from proteomics_revertant.ebayes import chi2_1_from_p
lam  = results["lambda_artifact"].iloc[0]
p_gc = stats.chi2.sf(chi2_1_from_p(results["dose_P"]) / lam, 1)
```

Before doing that, read the caveat in §2.1: λ on these panels is usually **below**
1, and dividing by a λ under 1 inflates every statistic rather than shrinking it.

#### Integration and verdict

| Column | Meaning |
|---|---|
| `sign_concordant` | `True` if the two corroborating contrasts move in the same direction. |
| `iut_P` | Intersection–union p-value: `max` of the two corroborating p-values. Tests "differs from the revertant **and** from baseline, in the same direction". |
| `iut_FDR` | Benjamini–Hochberg adjustment of `iut_P`. **This is the strict standard.** |
| `monotone_in_dose` | `True` if mean abundance moves in one direction across the three allele dosages. |
| `pa_trend_z` | Cochran–Armitage trend statistic for *detection* against allele dosage. Detection track only. |
| `pa_trend_P` | Exact permutation p-value for that detection trend, two-sided. |
| `pa_trend_FDR` | Benjamini–Hochberg adjustment, computed over the detection track only. |
| `pa_hit` | `True` if `pa_trend_FDR` < 0.05. |
| `artifact_flag` | `True` if any artefact contrast is significant at FDR < 0.05 for this feature — a warning that this feature is clone-unstable. |
| `call` | Single-word verdict, in priority order: `variant_attributable` (IUT FDR < 0.05 **and** monotone — the strict standard), `presence_absence_hit` (detection trend FDR < 0.05), `dose_trend_only` (dose FDR < 0.05 but corroboration failed), `filtered_no_model`, or blank. |

### 4.2 `<key>_overall.tsv` — the genome-wide answer

Forty-two fields. `<key>_overall.txt` states the same conclusions in prose.

**Design and calibration**

| Field | Meaning |
|---|---|
| `design`, `n_clones`, `residual_df` | Panel identity and size. `residual_df` = clones − 2. |
| `proteins_tested` | Features entering the genome-wide calculation. |
| `lambda_hat`, `lambda_lo`, `lambda_hi` | Genomic-control factor with a 95% bootstrap interval (§2.1). Unfloored. |
| `lambda_applied` | `max(lambda_hat, 1)`, used only as a conservative multiplier in the omnibus noise subtraction below. **Not** applied to any p-value. |
| `n_artefact_contrasts` | Matched-dosage clone pairs available. **Zero means λ is unmeasurable and every number here is an upper bound.** |
| `equivalence_resolution_log2` | The tightest difference that could be *declared* negligible for a typical feature. If this is 0.49, you cannot certify equivalence at 0.2 whatever the data show. |

**How much of the system responded**

| Field | Meaning |
|---|---|
| `pi1`, `pi1_lo`, `pi1_hi` | Storey's π₁ — estimated **fraction** of measured features genuinely responding to allele dosage, with bootstrap interval. Inferred from the whole p-value distribution, so it counts real effects too small to call individually. |
| `proteins_implicated` | `pi1 × proteins_tested`, rounded — the implied count. |
| `n_dose_FDR05` | Features with `dose_FDR` < 0.05. |
| `n_dose_p01` | Features with raw `dose_P` < 0.01. |

**How large the response is** (all in log2 units)

| Field | Meaning |
|---|---|
| `global_effect_log2_per_allele` | Root-mean-square true per-allele effect, with estimation error removed. |
| `global_effect_lo`, `global_effect_hi` | Bootstrap interval for the above. |
| `dose_span` | Range of the dosage axis, normally 2 (from 0 to 2 alleles). |
| `global_effect_full_range` | The per-allele effect × `dose_span` — effect across the full allelic range. |
| `global_effect_net_of_noise` | **The number to quote.** Per-allele effect after subtracting the editing-noise floor via the permutation null. |
| `global_effect_net_full_range` | The above × `dose_span`. |
| `noise_floor_tau2_on_dose_scale` | The editing-noise floor itself, as a variance on the dose scale. |
| `net_effect_artefact_method` | The same net effect derived from the artefact contrasts instead of the permutation null — an independent cross-check. |
| `noise_subtraction` | Which method produced the headline net effect. |

**The permutation test**

| Field | Meaning |
|---|---|
| `perm_P` | **The headline global p-value.** Based on the dispersion statistic, restricted to lineage-matched relabellings. |
| `perm_P_dispersion` | Identical to `perm_P`; named explicitly so the statistic is unambiguous. |
| `perm_P_tail` | The same test using a count of significant features instead of dispersion. **Reported for completeness — do not quote it.** Relabelling dosages also relabels editing lineage, which makes this statistic anticonservative; verified on true-null simulations. |
| `perm_P_unrestricted` | Without the lineage-matching restriction. Diagnostic only, for the same reason. |
| `perm_floor` | `1 / n_perms` — the smallest p-value this design can produce. **Essential context**: see §5. |
| `n_perms` | Relabellings in the restricted null. |
| `n_perms_total` | Relabellings before lineage restriction. |
| `n_distinct_relabellings` | Distinct relabellings that exist for this panel. |
| `perm_mode` | `exhaustive` (all enumerated) or `sampled`. |
| `lineage_matched` | Whether the lineage restriction was applied. |
| `tau2_observed` | Dispersion statistic at the true labelling. |
| `tau2_null_median` | Median of that statistic across the null. |
| `net_effect_permutation` | `√(tau2_observed − tau2_null_median)`, the net effect on the dose scale. |
| `observed`, `null_median`, `null_max` | The tail-count statistic observed, and its null median and maximum. Diagnostic. |

### 4.3 `<key>_qc_*.tsv` — seven diagnostics

| File | Columns | What to look for |
|---|---|---|
| `qc_missingness` | `library`, `pct_missing`, `clone`, `state`, `dose`, `plex` | One run with far more missingness than its peers is a candidate for exclusion. |
| `qc_mnar_profile` | `bin`, `mean_frac_missing`, `n_proteins` | Missingness should rise steeply at low intensity. A flat profile means dropout is *not* abundance-driven, which undermines the missing-not-at-random assumption. |
| `qc_artefact_scale` | `contrast`, `n`, `median`, `robust_sd`, `frac_abs_gt_0p5` | **The empirical noise floor.** `robust_sd` is the typical disagreement between clones that should be identical. `frac_abs_gt_0p5` is the fraction of features differing by more than 0.5 log2 for no genetic reason. |
| `qc_null_calibration` | `contrast`, `n`, `lambda_gc`, `robust_sd_of_t` | Per-contrast λ (§2.1). `robust_sd_of_t` is the same over-dispersion read off the t-statistics — roughly √λ when the null is well behaved. Disagreement between the two indicates a skewed rather than merely widened null. |
| `qc_directional_balance` | `contrast`, `n_nominal`, `n_up`, `n_down`, `frac_up`, `binomial_P`, `flag` | Two-sided tests should split hits evenly. A strong imbalance points to one clone carrying a coherent one-directional shift — the signature of a copy-number change. |
| `qc_coherent_blocks` | clone, `max_rolling_abs_deviation` | Rolling mean of each clone's deviation along the feature index. A sustained regional shift is a CNV mimic. Pair with directional balance. |
| `qc_clone_correlation` | clone × clone matrix | Complete-case correlation between clone mean profiles. A clone markedly less correlated with its peers has drifted. |

### 4.4 `<key>_pca_*` — how the libraries actually arrange themselves

PCA runs on complete cases only (no imputation), and the kept/dropped counts are
printed and written into the figure subtitle: a PCA on 60% of the features is a
different statement from one on 95%.

| File | Columns | Notes |
|---|---|---|
| `pca_scores.tsv` | `library`, `PC1`…`PC10`, `clone`, `state`, `dose`, `class`, `plex` | One row per run. Join on `class` to colour by genotype. |
| `pca_loadings.tsv` | `component`, `rank`, `protein_id`, `loading`, `abs_loading`, `pc_var_explained` | Top 15 features per component, ranked, for PC1–PC10. |
| `pca_influence.tsv` | `rank`, `protein_id`, `influence`, `influence_pct`, `dominant_component`, `loading_on_dominant`, `cumulative_influence_pct` | **Up to the top 100 features overall, ranked.** |
| `pca.png` | — | Scores for PC1v2, PC1v3, PC2v3, plus a scree panel. |
| `pca.txt` | — | Prose reading, including a warning if the revertant fails to return toward baseline. |

The influence score is variance-weighted, `Σⱼ loadingᵢⱼ² × varⱼ`. Squaring makes
the direction irrelevant — what is measured is how much variation a feature
accounts for — and weighting by each component's explained variance stops a large
loading on a negligible component from outranking a moderate loading on the
dominant one. Because loadings are unit-normalised, `influence_pct` reads
directly as a percentage of the retained variance, and
`cumulative_influence_pct` shows how concentrated the structure is: if the top 10
features reach 30%, a handful of features dominate the decomposition.

Only PC1–PC3 pairs are plotted; the tables still cover PC1–PC10, because a
component can matter for interpretation long after it stops being worth a
scatter plot.

> **Interpreting the figure.** With one clone per genotype, separation on some
> component is guaranteed — clone identity and genotype are the same thing. The
> informative question is the *ordering*: whether the revertant returns toward
> the baseline. If PC1 separates by plex or passage, or places the revertant
> further from wild type than the edit is, the dominant axis is technical and the
> per-feature results are carrying that load.

### 4.5 Provenance

Written once per invocation.

| File | Contents |
|---|---|
| `run_log.txt` | Start time (UTC), elapsed seconds, exact command, Python and library versions, platform, and per-dataset: clone count, residual df, features tested, λ with interval, output dimensions, and a **SHA-256 for every input file**. |
| `run_metadata.json` | The same content plus the full per-dataset metadata — prior degrees of freedom `d0`, prior variance `s0²`, consensus ICC, contrast list, call and track tallies. |
| `calibration_summary.tsv` | `dataset`, `n_clones`, `residual_df`, `proteins_tested`, `n_artefact_contrasts`, `lambda_hat`, `lambda_lo`, `lambda_hi`, `lambda_applied`. |

Inputs are hashed rather than merely named because a dataset regenerated in place
is otherwise indistinguishable from one that was not. Library versions are
recorded because the moderated variances depend on scipy's special functions and
shift in the last digits between releases — a result that will not reproduce
exactly is usually a version difference rather than a bug.

λ is deliberately surfaced at run level as well as per dataset. It is not
comparable to a fixed threshold in isolation; the useful view is across
datasets, where a panel whose λ sits well above its neighbours' is the one whose
standard errors are least trustworthy.

## 5. Two limits worth stating before reading a result

**The global permutation p-value has a floor.** Five clones admit only 30
distinct dosage relabellings, and 14 after restricting to those with comparable
lineage leverage, so the smallest achievable global p-value is 0.071. When the
summary reports P at the floor it means the effect is as extreme as the design
can register — not that it failed. Fifteen clones move the floor to 0.017 and
twenty-five to 0.012, at which point the p-value is a measurement rather than a
boundary. Always read `perm_P` next to `perm_floor`.

**Individual features are on firmer ground than the genome-wide claim.** A
feature with a large, monotone, revertant-confirmed dose response is well
supported even when the global test sits at its floor. The two questions have
different sample sizes: a thousand features against five to twenty-five clones.

## 6. What replication buys

Five shipped panels, identical planted truth and seed. Technical replication is
*lower* in the replicated panels (2 runs per clone rather than 4); every gain
below comes from independently derived clones.

| Panel | Clones | Runs | df | SE of the −2.2 effect | Artefact contrasts | Permutation floor | Global P | Sensitivity |
|---|---|---|---|---|---|---|---|---|
| `A_homozygous_wt` | 5 | 20 | 3 | 0.185 | 2 | 0.0714 | 0.0714 *(at floor)* | 0.19 |
| `A_replicated_x3` | 15 | 30 | 13 | 0.112 | 12 | 0.0172 | 0.0172 | 0.41 |
| `A_replicated_x5` | 25 | 50 | 23 | 0.091 | 22 | 0.0119 | 0.0238 *(off floor)* | 0.45 |
| `B_heterozygous_wt` | 5 | 20 | 3 | — | 2 | 0.0333 | 0.0667 | — |
| `B_heterozygous_wt_minimal` | 3 | 12 | 1 | — | 0 | 0.1667 | 0.3333 | — |

The standard error falls as 1/√(clones): 0.185/0.091 = 2.03 against an expected
√5 = 2.24. Sensitivity more than doubles. The λ interval narrows from 0.17 to
0.10 wide. The minimal panel has no two clones sharing a dosage, so λ is
unmeasurable and the pipeline refuses to make a calibrated claim at all, marking
every number as an upper bound.

Once clone variance is non-zero, the standard error of a genotype contrast is
floored by τ²/n_clones regardless of how often each clone is run. Clones buy
inference; injections do not.

One quantity does not improve with replication. The **equivalence resolution** —
the tightest bound within which a typical feature can be *declared* unchanged
between two clones — stays at 0.49–0.67 log2 across all panels, because each
pairwise comparison is still a difference of two clone means. The aggregate
revertant-versus-baseline contrast does improve (0.384 → 0.322 log2 from 15 to 25
clones) because it averages over clones. Certifying that a revertant matches wild
type to within 0.2 log2 is not achievable in any of these panels, and the summary
says so rather than presenting a non-significant difference as equivalence.

## 7. Validation

`python -m proteomics_revertant.tests` runs 37 checks. Each computes an
independent reference — brute force, Monte Carlo, or a closed form derived a
different way — rather than comparing the code to itself. Selected results:

- Empirical Bayes prior recovered from 200k draws of a known scaled-F hierarchy:
  d0 = 8.15 (true 8.0), s0² = 0.2504 (true 0.25).
- Exact detection trend test against 40,000-draw Monte Carlo: 0.00021 vs 0.00020,
  0.02492 vs 0.02500, 0.23519 vs 0.23505.
- Batched fit against an explicit per-feature loop: maximum difference 4e-15 to
  9e-15 across five panels, 2–13× faster.
- Two-sidedness: reversing the dosage axis negates every logFC and leaves every
  p-value unchanged to 4.4e-13.
- Zero false discoveries at FDR < 0.05 across eight true-null datasets, on every
  track, with raw p-values calibrated at 0.068–0.078 against a nominal 0.05.
- Accelerated kernels reproduce the reference fit to 1e-9, and agree between
  numpy, torch-CPU and torch-CUDA to 8.9e-16.

Two regression tests pin failures found during development: treating a recessive
feature's lack of fit as hypervariance (which tripled its standard error and lost
a real hit), and saturating the error term (which produced tens of false
discoveries on null data).

## 8. Data format

A dataset is three files sharing a prefix, with an optional HDF5 mirror:

```
<key>.intensities.tsv   features x runs, log2 intensity, empty cell = not detected
<key>.samples.tsv       one row per run: library, clone, state, dose, lineage, tech_rep, plex
<key>.manifest.json     name, units, provenance
<key>.truth.tsv         optional answer key, synthetic data only
```

`dose` must be 0, 1 or 2. `datasets.py: validate` rejects malformed inputs with
specific messages, and `design_from_samples` reconstructs the clone panel from
`samples.tsv` alone. A hand-written four-clone dataset with names the code has
never seen is exercised in the test suite.

The HDF5 mirror requires `h5py`, which is not in `requirements.txt`; without it
the `--hdf5` flag is unavailable and the corresponding test skips cleanly.

**One edit per line.** The model carries a single ordinal dose axis. A line
carrying two independent edits cannot currently be represented — it requires a
second dose column — and such designs must be run as separate datasets.

## 9. Applying this to real data

The example datasets are synthetic, with a known answer key, so the pipeline can
be validated end to end. Nothing about the analysis is specific to them. For real
experiments:

- Confirm zygosity in the assay itself before trusting the dosage axis:
  allele-specific expression at the locus, NMD of the target transcript, and
  sequencing of the revertant allele for silent scars.
- Check for clone-specific copy number. `_qc_coherent_blocks.tsv` flags a clone
  with a sustained regional shift and `_qc_directional_balance.tsv` flags a
  contrast whose hits are lopsided; together those are the signature of a
  culture-adaptation CNV rather than biology.
- For proteomics, prefer peptide-level modelling (`msqrob2`, `MSstats`) or an
  explicit dropout model (`proDA`) over the clone-mean reduction used here. The
  dosage axis, the intersection–union structure and the matched-dosage
  calibration all carry over directly.
- For transcriptomics there is no missingness in this sense — a zero is an
  observed count. Use `filterByExpr` on the full design and a negative-binomial
  fit; do not impute counts.

## 10. Module map

| File | Contents |
|---|---|
| `config.py` | Clone panels, allele dosages, lineages, simulation constants, `replicated_design(n)` |
| `datasets.py` | On-disk format, loader, validation, design inference from `samples.tsv` |
| `make_data.py` | Generates the example datasets and their answer keys |
| `simulate.py` | Intensity generation: clone, lineage and plex effects, MNAR dropout, a planted CNV |
| `ebayes.py` | `squeezeVar` / `fitFDist` / `trigammaInverse` ports, Benjamini–Hochberg, genomic-control λ |
| `fastfit.py` | Batched weighted least squares, grouped by observation-count vector |
| `analysis.py` | Clone-level model, contrasts, intersection–union test, detection track, TOST |
| `pca.py` | Separate command: genotype-coloured PCA, clouds, loading and influence tables |
| `omnibus.py` | λ, π₁, net effect, equivalence resolution, permutation test, plain-language summary |
| `columns.py` | Data dictionary generated from the design; writes `<key>_columns.md` and `.tsv` |
| `qc.py` | Missingness, MNAR profile, artefact scale, null calibration, directional balance, coherent blocks |
| `run.py` | CLI, provenance and run log |
| `backend.py` | Array-namespace resolution (`numpy` / `cupy` / `torch`), device memory chunk planning |
| `accel.py` | Closed-form p=2 weighted fit and `sweep_designs`, the resident permutation sweep |
| `bench.py` | Scaling benchmark: per-feature loop vs batched vs sweep, plus the `--panel` grid |
| `tests.py` | 37 checks |
