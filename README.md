# Allelic-series analysis for edited / revertant clone panels

Differential abundance analysis for experiments in which a variant is introduced
into a cell line and then reverted, with proteomics or transcriptomics readout.
The pipeline estimates the per-allele effect of the variant, calibrates it
against the experiment's own editing noise, and reports both per-feature results
and a single proteome-wide answer.

Pure Python (`numpy`, `scipy`, `pandas`). No R, no compiled extensions. The
`limma` functions it depends on are reimplemented in `ebayes.py` and validated
against their published definitions.

```bash
pip install -r requirements.txt
python -m proteomics_revertant.make_data --outdir data --hdf5   # synthetic examples
python -m proteomics_revertant.run  --data data --outdir results
python -m proteomics_revertant.tests                            # 31 checks
```

Point `--data` at your own directory of datasets to replace the examples. The
clone panel, allele dosages, lineages and plexes are read from `samples.tsv`;
nothing about the experimental design is hard-coded.

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
`limma::duplicateCorrelation`.

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

**Two-sided throughout.** Every reported p-value tests against zero in either
direction. The only one-sided quantities are the omnibus permutation p-values,
whose statistics are non-negative dispersion measures, as for an F-test.

**Intersection–union calls.** A feature is called variant-attributable only if it
differs from both the revertant and the baseline, in the same direction, and is
monotone in dosage. `max(p₁, p₂)` is a valid p-value for that intersection, so
Benjamini–Hochberg is applied once across features with no further correction.

**Calibration.** The matched-dosage clone pairs contain no variant signal by
construction, so the spread of their moderated t-statistics audits the standard
errors directly. λ is reported with a bootstrap interval; when λ > 1 the pipeline
also emits recalibrated p-values.

`METHODS.md` maps each of these to the published method it implements, states
where the implementation departs from standard practice and why, and lists the
recommended procedures that are *not* implemented.

## 3. Output

| File | Question it answers |
|---|---|
| `<key>_overall.txt` | Did the edit change the proteome at all, and by how much? |
| `<key>_results.tsv` | Which individual features changed. One row per feature. |
| `<key>_columns.md` | What every column means, with the formula behind it. |
| `<key>_qc_*.tsv` | Missingness, MNAR profile, noise scale, calibration, directional balance, coherent blocks, clone correlation. |
| `<key>_benchmark.txt` | Recovery against the answer key, for datasets that ship one. |

### The five numbers in the overall summary

**λ** — an audit of the error bars, estimated from clone pairs that share a
genotype. Near 1.0 means the standard errors are honest; above 1.1 they are
optimistic and the recalibrated columns should be used instead.

**π₁** — the estimated fraction of measured features genuinely responding to
allele dosage. Inferred from the whole p-value distribution, so it counts real
effects too small to call individually.

**Net effect** — the typical per-allele magnitude of that response in log2, after
subtracting the editing-noise floor via the permutation null. This is the number
to quote.

**Editing-noise floor** — the same magnitude computed on matched-dosage clone
pairs. An effect not clearly larger than this is inside the noise of the panel.

**Permutation P** — reassigns which clone carries which allele dosage and refits
everything. Clones stay intact; only the clone-to-dosage link is broken.

## 4. Two limits worth stating before reading a result

**The global permutation p-value has a floor.** Five clones admit only 30
distinct dosage relabellings, and 14 after restricting to those with comparable
lineage leverage, so the smallest achievable global p-value is 0.071. When the
summary reports P at the floor it means the effect is as extreme as the design
can register — not that it failed. Fifteen clones move the floor to 0.017 and
twenty-five to 0.012, at which point the p-value is a measurement rather than a
boundary.

**Individual features are on firmer ground than the proteome-wide claim.** A
feature with a large, monotone, revertant-confirmed dose response is well
supported even when the global test sits at its floor. The two questions have
different sample sizes: a thousand features against five to twenty-five clones.

## 5. What replication buys

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
√5 = 2.24. Sensitivity more than doubles. The λ interval narrows from 0.19 to
0.09 wide. The minimal panel has no two clones sharing a dosage, so λ is
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

## 6. Validation

`python -m proteomics_revertant.tests` runs 31 checks. Each computes an
independent reference — brute force, Monte Carlo, or a closed form derived a
different way — rather than comparing the code to itself. The full table is in
`METHODS.md` §4. Selected results:

- Empirical Bayes prior recovered from 200k draws of a known scaled-F hierarchy:
  d0 = 8.15 (true 8.0), s0² = 0.2504 (true 0.25).
- Exact detection trend test against 40,000-draw Monte Carlo: 0.00021 vs 0.00020,
  0.02492 vs 0.02500, 0.23519 vs 0.23505.
- Batched fit against an explicit per-protein loop: maximum difference 4e-15 to
  9e-15 across five panels, 2–13× faster.
- Two-sidedness: reversing the dosage axis negates every logFC and leaves every
  p-value unchanged to 4.4e-13.
- Zero false discoveries at FDR < 0.05 across eight true-null datasets, on every
  track, with raw p-values calibrated at 0.068–0.078 against a nominal 0.05.

Two regression tests pin failures found during development: treating a recessive
feature's lack of fit as hypervariance (which tripled its standard error and lost
a real hit), and saturating the error term (which produced tens of false
discoveries on null data).

## 7. Data format

A dataset is three files sharing a prefix, with an optional HDF5 mirror:

```
<key>.intensities.tsv   features x runs, log2 intensity, empty cell = not detected
<key>.samples.tsv       one row per run: library, clone, state, dose, lineage, tech_rep, plex
<key>.manifest.json     name, units, provenance
<key>.truth.tsv         optional answer key, synthetic data only
```

`datasets.py: validate` rejects malformed inputs with specific messages, and
`design_from_samples` reconstructs the clone panel from `samples.tsv` alone. A
hand-written four-clone dataset with names the code has never seen is exercised
in the test suite.

## 8. Applying this to real data

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
  calibration all carry over directly. See `METHODS.md` §3.
- For transcriptomics there is no missingness in this sense — a zero is an
  observed count. Use `filterByExpr` on the full design and a negative-binomial
  fit; do not impute counts.

## 9. Module map

| File | Contents |
|---|---|
| `config.py` | Clone panels, allele dosages, lineages, simulation constants, `replicated_design(n)` |
| `datasets.py` | On-disk format, loader, validation, design inference from `samples.tsv` |
| `make_data.py` | Generates the example datasets and their answer keys |
| `simulate.py` | Intensity generation: clone, lineage and plex effects, MNAR dropout, a planted CNV |
| `ebayes.py` | `squeezeVar` / `fitFDist` / `trigammaInverse` ports, Benjamini–Hochberg |
| `fastfit.py` | Batched weighted least squares, grouped by observation-count vector |
| `analysis.py` | Clone-level model, contrasts, intersection–union test, detection track, TOST |
| `pca.py` | separate command: genotype-coloured PCA, clouds, loading highlights |
| `omnibus.py` | λ, π₁, net effect, equivalence resolution, permutation test, plain-language summary |
| `columns.py` | Data dictionary generated from the design; writes `<key>_columns.md` and `.tsv` |
| `qc.py` | Missingness, MNAR profile, artefact scale, null calibration, directional balance, coherent blocks |
| `run.py` | CLI |
| `tests.py` | 31 checks |
