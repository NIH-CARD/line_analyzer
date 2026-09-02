# CLAUDE.md

Context for working on this repository.

## What this is

A statistics pipeline for CRISPR edit / revertant clone panels, in proteomics or
transcriptomics. Given a set of clones carrying 0, 1 or 2 copies of a variant --
including revertants, where the edit was reversed -- it estimates the per-allele
effect on every protein, and separately answers whether the edit perturbs the
proteome at all.

Pure Python: `numpy`, `scipy`, `pandas`, `matplotlib`. No R, no rpy2, no limma,
no statsmodels, no compiled extensions. `ebayes.py` reimplements the parts of
limma that are needed (`trigammaInverse`, `fitFDist`, `squeezeVar`).

## Commands

```bash
python -m proteomics_revertant.make_data --outdir data --hdf5   # regenerate fixtures
python -m proteomics_revertant.run  --data data --outdir results   # ~110 s
python -m proteomics_revertant.pca  --data data --outdir results   # ~5 s
python -m proteomics_revertant.columns --data data --outdir results
python -m proteomics_revertant.tests                               # 37 checks, ~4 min
```

`tests.py` is a self-contained runner, not pytest. Every check computes an
independent reference -- brute force, Monte Carlo, or a closed form derived a
different way -- rather than comparing the code to itself. Keep it that way.

`run.py` writes three run-level files alongside the per-dataset outputs:
`run_log.txt` (prose provenance -- versions, platform, command, elapsed, and a
SHA-256 per input file), `run_metadata.json` (the same, machine-readable, plus
`d0`, `s0^2` and the consensus ICC per dataset) and `calibration_summary.tsv`
(lambda and design size for every dataset in the run). Input files are hashed
rather than merely named because a dataset regenerated in place is otherwise
indistinguishable from one that was not.

## The one thing to understand before changing anything

**The clone is the unit of replication, not the library.** Technical replicates
are collapsed to clone means and residual df is `(clones - parameters)`, which is
3 for a 5-clone panel. Any change that makes the residual df look like the
library count has reintroduced pseudoreplication and every p-value in the output
becomes wrong by orders of magnitude. `t_df` guards this.

The second thing: **clones of identical allele dosage but different editing
history are the error term.** Wild type and the homozygous revertant both carry
zero alleles; whatever separates them is drift and off-target editing. Because
two clones sit at dosage 0 and two at dosage 1, the fitted dose line cannot pass
through all five points, and the residual *is* clone-to-clone noise. Editing
noise is the ruler, not a correction applied afterwards.

## Decisions that were tried and rejected -- do not "fix" these back

Each of these looks like an improvement and is not. All are pinned by tests.

1. **Error term from the dose-saturated model.** Recovers the planted effect
   beautifully (FDR 6e-10 vs 0.001) but with five clones the dominance term also
   absorbs genuine clone artefacts, the residual collapses, and null datasets
   start producing *dozens* of discoveries at FDR<0.05. The error term stays
   additive. See the comment block in `analysis.py`.

2. **Variance-outlier test on the additive residual.** A recessive protein has a
   large additive residual purely from lack of fit; flagging it as hypervariable
   triples its SE and buries a real hit. The hypervariability test runs on the
   *saturated* residual while the error term stays additive. `t_lackoffit` pins
   this.

3. **Unrestricted permutation null for the omnibus test.** Relabelling allele
   dosages also relabels which clones share an editing lineage, and a
   lineage-shared off-target set loads onto some dosage vectors more than
   others. Measured on 30 true-null simulations: unrestricted fires at 10% for a
   nominal 5% test. The null is restricted to lineage-matched relabellings. Cost
   is a coarse floor (0.071 for the 5-clone panel), which is reported.

4. **Tail-count statistic as the headline permutation p-value.** Same lineage
   problem, worse. It is computed and reported but explicitly marked "do not
   quote"; the dispersion statistic (`tau2`) is primary.

5. **Any imputation, anywhere.** Especially class-wise median imputation, which
   preserves the between-group difference while shrinking within-group variance
   and so pushes p-values down in a self-fulfilling way. Proteins absent from a
   whole genotype go to the detection-count track instead. `t_no_imputation`
   scans the source for `fillna` and friends.

6. **Percentage-missing filters.** They delete precisely the on/off proteins a
   knockout produces. Filtering is on counts: at least 2 observations in at least
   one genotype state.

7. **Computing lambda on `dose_P`.** This is what a GWAS does, and it is wrong
   here. A GWAS assumes most markers are null; this design *provides* a null in
   the matched-dosage artefact contrasts, which is a stronger footing. Calibrating
   on `dose_P` would fold genuine variant signal into the constant and shrink the
   real effects the experiment exists to measure. `t_genomic_lambda` pins the
   estimator to the artefact contrasts.

8. **Robust SD^2 of the artefact t-statistics as lambda.** The original
   estimator. It is a reasonable measure of over-dispersion but it is not the
   genomic-control factor, and it was labelled `lambda_gc` in the QC output while
   not being one. It also has no fixed reference: the spread of a t-statistic
   depends on the degrees of freedom, so the number was not comparable between a
   3-df panel and a 23-df one. Lambda is now `median(chi2_1^-1(1-p)) /
   median(chi2_1)` computed from p-values, which is df-free because p is uniform
   under any null. The robust SD is still reported alongside it in
   `_qc_null_calibration.tsv` as a secondary diagnostic -- a disagreement between
   the two means the null is skewed rather than merely widened.

9. **Selecting artefact contrasts by `Contrast.role`.** `role == "artefact"` also
   matches the aggregate `revertant_vs_baseline`, which re-uses the same clones
   as the pairwise contrasts and carries an averaged standard error on a
   different scale. `analyse()` used the role while `omnibus` and `qc` used the
   `artifact_` name prefix, so the same run reported lambda = 1.0 in
   `_results.tsv` and 1.057 in `_overall.tsv`. There is now ONE selector,
   `ebayes.artefact_p_columns`, used by all three; `t_genomic_lambda` asserts
   they agree.

10. **Applying genomic control inside the pipeline.** Lambda is computed and
    reported in four places; it is applied in none. There is deliberately no
    `dose_P_recalibrated` column -- it existed and was removed. Two reasons.
    Lambda on these panels is usually BELOW 1, and genomic control *divides* by
    lambda, so a column labelled "recalibrated" would have been inflating every
    statistic while sounding conservative. And a corrected column invites
    quoting without the reader ever deciding whether correction was warranted;
    that decision belongs to the analyst, who can apply it in one line from the
    reported lambda. `t_genomic_lambda` asserts no `*recalibrat*` column exists
    and that `dose_P` fits an uncorrected reconstruction better than any
    lambda-corrected one.

    Note the one place lambda still multiplies something: `omnibus` uses
    `max(lambda, 1)` in the tau^2 noise subtraction,
    `tau2 = mean(beta^2) - lambda * mean(SE^2)`. That is a conservative
    effect-size floor, not a p-value correction, and it is a no-op wherever
    lambda <= 1 -- which is four of the five shipped panels.

## Known honest limitations -- these are findings, not bugs

- Raw dose p-values run ~0.077 against a nominal 0.05 under the null. Heavy-tailed
  clone artefacts meeting 3 residual df. Does **not** propagate to reported calls:
  0 false discoveries at FDR<0.05 across null datasets.

- **Lambda does not converge to 1 as the panel grows, and it should not be made
  to.** The obvious reading of the shipped panels -- lambda 0.928 at 5 clones,
  0.909 at 15, 0.847 at 25 -- is that this is a small-panel artefact that more
  clones would cure. It is not. Measured on true-null simulations, three seeds
  per size, `replicated_design(n)`:

  | clones | lambda (median) | P(p<0.05) | P(p<0.01) | P(p<0.001) |
  |---|---|---|---|---|
  | 5 | 1.018 | 0.0073 | 0.0000 | 0.0000 |
  | 15 | 0.939 | 0.0380 | 0.0038 | 0.0000 |
  | 25 | 0.825 | 0.0392 | 0.0070 | 0.0003 |
  | 50 | 0.769 | 0.0418 | 0.0123 | 0.0024 |
  | 100 | 0.876 | 0.0507 | 0.0162 | 0.0046 |
  | *nominal* | *1.000* | *0.0500* | *0.0100* | *0.0010* |

  Lambda settles in the 0.77-0.88 band and stays there. What is actually
  happening is a **heavy-tailed null**: the bulk of the artefact-contrast
  distribution is NARROWER than the t reference (lambda < 1) while the deep tail
  is FATTER (P(p<0.001) reaching 4.6x nominal at 100 clones). Those are the same
  phenomenon read at different quantiles, and they move in opposite directions,
  which is why no single scale factor fixes both.

  Two things follow, and both matter.

  *Lambda must stay floored at 1.* Genomic control divides chi-square by lambda,
  so applying a lambda of 0.87 would multiply every statistic UP and make the
  already-anticonservative deep tail substantially worse. `max(lambda, 1)` is
  not timidity; it is the only safe direction when the null is heavy-tailed.

  *Small panels are conservative everywhere, not well calibrated.* At 5 clones
  lambda is 1.018 and looks perfect, but the tail rate is 0.0073 against a
  nominal 0.05 -- the whole distribution is squeezed by moderation borrowing
  heavily against 3 residual df. The lambda near 1 there is a coincidence of the
  median, not evidence of calibration. Do not read a 5-clone lambda as
  reassurance.

  The trend with clone count is moderation fading, not calibration degrading:
  recomputing the same p-values on the unmoderated residual df gives lambda
  0.870 at 5 clones and 0.875 at 100 -- essentially flat. Moderation is what
  lifts the small-panel lambda to 1.018, and its influence shrinks as
  `df_residual` grows past `d0` (~6-8). The underlying ~0.87 was always there.

  Ruled out as causes: the consensus-ICC weighting (features with balanced,
  complete observation counts give lambda 0.844 against 0.848 for ragged ones at
  25 clones, so constant weights change nothing) and the empirical-Bayes step
  itself (moderated and unmoderated converge above ~25 clones).
- The global permutation p-value has a floor of ~0.071 on a 5-clone panel. When
  the summary reports P at the floor it means the effect is as extreme as the
  design can register, not that it failed.
- The simulation plants a one-directional +0.42 CNV on 90 proteins in the
  highest-leverage clone. Under the null this makes 68% of nominal hits go up.
  That is deliberate; `_qc_directional_balance.tsv` and `_qc_coherent_blocks.tsv`
  are the detectors and both fire correctly.
- **One edit per line.** The model carries a single ordinal dose axis: `Clone`
  holds one scalar `dose`, `validate` hard-rejects anything outside {0,1,2}
  (`datasets.py`), and `build_design` emits `[intercept, dose]` plus an optional
  dominance term. A line carrying two independent edits cannot be represented --
  it needs a second dose column, and today those must be run as separate
  datasets. The *numerics* are not the obstacle: `accel.fit_general` and
  `fastfit.fit_batched` are both general-p, and `fit_batched` already groups by
  observation-count vector. What a second edit would touch is the design layer --
  `SAMPLE_COLUMNS` and `validate`, `build_design`, `contrasts_for`, joint
  relabelling in `omnibus.dose_permutations`, and `columns.data_dictionary`.
  Note that two edits are only *separable* if the panel breaks their
  collinearity: lines carrying A alone, B alone, and both. No amount of design
  code rescues a panel where the two edits always travel together.

## Cleanup backlog

Roughly in order of value.

- [ ] **Performance: the fit is memory-bound at ~1.3 FLOP/byte.** Measured on
      one throttled core: the per-protein loop runs 7-40x slower than the
      batched einsum, and a closed-form p=2 kernel (explicit 2x2 inverse, no
      `solve`) is a further 2.7-4.9x on top. `consensus_icc` is still a
      per-protein loop -- 30,597 `np.linalg.lstsq` calls per dataset. Also hoist
      `clone_summaries` out of the permutation loop: clone means and counts do
      not depend on allele dosage, so 31 permutations recompute an invariant.
      `bench.py` measures all of this; run `python -m proteomics_revertant.bench`.
      GPU is not worth it -- see the benchmark's own conclusion and the notes
      below.
- [ ] **`analyse()` is 243 lines.** Decompose into: clone reduction -> filtering
      and track assignment -> weighted fit -> moderation -> contrasts ->
      integration -> calls. Each step already has a comment header marking the
      seam. Do this one carefully; it is the function every test depends on.
- [ ] **`omnibus.omnibus()` is 129 lines and `narrate()` is 101.** Split the
      computation from the prose rendering; `narrate` should take a dataclass,
      not a dict of 30 keys.
- [ ] **Design/contrast logic is duplicated** between `analysis.contrasts_for`
      and `columns.data_dictionary`. They are kept in sync only by `t_columns`.
      Consider having the dictionary consume the `Contrast` objects directly
      rather than reconstructing the ordering by hand.
- [ ] **Test suite is ~4 minutes**, dominated by two checks at 112 s and 99 s
      (the CLI end-to-end and the replication comparison). Add a fast/slow split
      so the inner loop is seconds. Do not delete the slow ones.
- [ ] **Lint baseline: 30 findings, all cosmetic.** `ruff` is configured in
      `pyproject.toml` but the code has never been run through it. Breakdown:
      11 unsorted-imports, 5 redefined-while-unused (mostly test-local imports
      shadowing module-level ones), 3 semicolon statements, 3 unused variables,
      3 non-PEP585 annotations, 2 deprecated typing imports, 2 unused imports,
      1 if-else that should be a ternary. 18 auto-fixable with `ruff check
      --fix`. Deliberately left un-fixed so the diff is yours to review rather
      than buried in the baseline commit. Type annotations are inconsistent
      (`analysis.py` has almost none, `omnibus.py` has some).
- [ ] **README is ~4,000 words** and mixes user guide, statistical rationale and
      verification log. Split into `README.md` (how to run), `STATISTICS.md`
      (why), `VALIDATION.md` (the test evidence table).
- [ ] `simulate.py` is imported by `tests.py` for null-calibration checks even
      though the pipeline reads datasets from disk. That coupling is fine but
      undocumented; make it explicit.
- [ ] `pca.t_pca_figure` asserts the PC1-3 restriction by scraping source with
      `inspect.getsource`. Replace with a module-level `SCORE_PAIRS` constant
      that both the figure and the test import.
- [ ] **`analysis.py` and `omnibus.py` both compute lambda.** `analyse()` emits
      it as a per-row diagnostic column; `omnibus()` needs it with a bootstrap
      interval. They share `ebayes.artefact_p_columns` and `genomic_lambda` so
      the numbers agree, but the median-across-contrasts step is still written
      twice. Fold it into one function returning both.
- [ ] **README references `METHODS.md` three times (§3, §6, §7) and the file does
      not exist.** Either write it or fold those pointers into the
      `STATISTICS.md` / `VALIDATION.md` split above -- do not leave three dangling
      cross-references in the user-facing document.
- [ ] `accel.tau2_statistic` is dead code: `sweep_designs` inlines the same
      arithmetic. Either route the sweep through it or delete it, so there is one
      definition of the statistic rather than two that can drift apart.
- [ ] `h5py` is not in `requirements.txt`, so `t_hdf5` self-skips and
      `make_data --hdf5` fails on a clean install. It is genuinely optional --
      either add it as an extra or say so in the README.
- [ ] `sweep_designs(unscaled_var=...)` is only correct when every protein shares
      the same weights. Under ragged missingness `(X'WX)^-1` varies per protein
      and the scalar factor silently gives wrong standard errors. Documented in
      the docstring now; no caller passes it, but it is a live trap.

## GPU / scaling layer -- START HERE

`backend.py` resolves an array namespace (`numpy` / `cupy` / `torch`, CPU or
CUDA) and sizes chunks against free device memory. `accel.py` holds the kernels:
a closed-form p=2 weighted fit (explicit 2x2 inverse, no batched `solve`), a
general-p fallback, and `sweep_designs`, which drives P design matrices against
one resident copy of the clone summaries.

Two wins, and only the second needs a GPU:

* **Hoisting.** Clone means, counts and weights do not depend on allele dosage;
  only the design matrix does. The reference permutation test recomputes them
  for all 30 relabellings. Measured speedup from hoisting alone, on CPU:
  **11-311x** across the five shipped datasets.
* **Residency.** With the summaries on the device, a sweep touches them P times
  with no host round trip. This is the only shape in the pipeline with enough
  parallel work for a GPU.

Verified: closed form == general == per-protein `lstsq` to 1e-9; the sweep
statistic matches an explicit loop to 1e-10; chunking is exact; float32 agrees
to 0.00% on the sweep statistic. Cross-backend, numpy vs torch-CPU vs torch-CUDA
agree to 8.9e-16 on the statistic and 2.8e-14 on the per-protein coefficients.

**`backend="auto"` is sized, not just probed.** Picking a GPU merely because one
exists is wrong for this kernel: it is memory-bound, so a small sweep finishes
before the PCIe round trip and the CUDA context have paid for themselves. On the
shipped datasets (1000 proteins x 5-25 clones, i.e. 5k-25k elements) CUDA ran
7-25x *slower* than numpy and very nearly erased the hoisting win the fast path
exists to deliver -- `t_accel_permutation` caught it as "fast path is not faster:
1.1x", but only on a machine that has a GPU. `accel.GPU_MIN_ELEMENTS` (250k
elements, measured) now steers `auto`; an explicit backend is always honoured,
because `bench.py` has to be able to ask for the slow one on purpose.

The crossover, 30 designs, float64, RTX 5080 vs numpy:

| elements (proteins x clones) | CPU ms | GPU ms |
|---|---|---|
| 25,000 | 4.5 | 24.6 |
| 125,000 | 19.8 | 37.6 |
| 250,000 | 98.7 | 46.4 |
| 2,500,000 | 1447 | 279 |

**Where the GPU does pay.** A large panel -- hundreds of edited lines across tens
of thousands of analytes, the shape an agriculture variety trial produces -- sits
far above the crossover. Measured 12-36x at 10k-40k analytes x 100-400 lines,
agreeing with the CPU to 1e-16. Two statistical bonuses come with that panel size
and are worth knowing: the permutation floor collapses (0.071 on a 5-clone panel
to ~0.005 at `max_perms=200`), and residual df goes from 3 to several hundred, so
the heavy-tail fragility under "Known honest limitations" largely stops biting.

### THE FIRST TASK

`permutation_tau2_fast` is **opt-in and not yet a drop-in replacement.** The
batched kernel needs one design per protein, so it currently uses only proteins
observed in *every* clone. That drops 23-163 proteins per dataset, and where
the drop is large the p-value moves:

| dataset | dropped | reference P | fast P |
|---|---|---|---|
| A_homozygous_wt | 28 | 0.0714 | 0.0714 |
| A_replicated_x3 | 128 | 0.0172 | 0.0172 |
| **A_replicated_x5** | **163** | **0.0238** | **0.0119** |
| B_heterozygous_wt | 23 | 0.0667 | 0.0667 |
| **B_heterozygous_wt_minimal** | -1 | **0.3333** | **0.1667** |

The fix: group proteins by observation-count vector and run one sub-design per
group. `fastfit.fit_batched` **already does exactly this** for the main fit --
port that grouping into `accel.sweep_designs`. `t_accel_permutation` pins the
current divergence to those two datasets, so it will fail the moment the set
changes in either direction. Do not wire the fast path in as the default until
that test reports 5/5.

Empirical-Bayes moderation is already applied in the fast path, matching
`analyse()`; that was checked and is not the cause of the divergence.

### Benchmarking

```bash
python -m proteomics_revertant.bench                       # CPU sweep
python -m proteomics_revertant.bench --backend cupy        # CUDA
python -m proteomics_revertant.bench --backend torch --device cuda --dtype float32
python -m proteomics_revertant.bench --accel-only          # just the sweep shape

# analyte x edit x replicate grid, capped at 40k analytes in 5k steps
python -m proteomics_revertant.bench --panel --backend torch --device cuda --verify
python -m proteomics_revertant.bench --panel --edits 100,200,400 --reps 3
```

`--panel` is the large-panel shape: "analyte" rather than "protein", because the
kernel does not care whether a row is a protein group or a transcript. It times
the reduction and the sweep separately, since the two axes cost time in different
places -- replicates widen only the raw sample matrix and are paid **once** in
the reduction to line means, while edits widen every design matrix and are paid
**P times** in the sweep. `--verify` recomputes every grid point on numpy and
reports the deviation; across the 96-point default grid the worst was 1.8e-15.

The `complete` column is worth watching: it counts analytes observed in every
line, which is the population the batched path can currently use (see THE FIRST
TASK). Ragged missingness scales badly with line count at one replicate -- at 2%
missing, 40 lines drops 54% of analytes -- but three replicates brings that to
0.06%. Since replication is standard in a variety trial, the ragged-missingness
limitation binds far less on large panels than on the 5-clone ones.

Measured on one throttled core, float64: the kernel is memory-bound at
**1.25-1.42 FLOP/byte** at every size, so time = bytes / bandwidth and a GPU
only helps when the data is resident. A 30-design sweep at 1M proteins x 25
clones takes 30 s here; the projection for an A100 is well under a second.
Below roughly 100k proteins the PCIe round trip costs more than the compute.

## Fast start

```bash
ruff check .                       # 30 cosmetic findings, 18 auto-fixable
ruff check . --fix                 # start here; it is a safe first commit
python -m proteomics_revertant.tests   # must stay 37/37
```

The `analyse()` decomposition is the highest-value item but also the riskiest;
do the lint pass and the `SCORE_PAIRS` constant first to get a feel for the test
suite's coverage before touching it.

## Repository hygiene

**This repo has been edited concurrently by more than one agent session.** Three
times, modules appeared or changed between turns with no record: `datasets.py` +
`make_data.py`, then `fastfit.py`, then the TOST equivalence columns and two
replicated datasets. Each change turned out to be sound, but each also broke
tests that had to be tracked down after the fact.

Git is now initialised for exactly this reason. **Commit before and after any
work session**, so the next person can diff instead of doing forensics on file
mtimes.

A fourth incident, and the reason the layout note below matters: the working tree
was once found holding only 5 of the 16 modules, flattened to the top level with
no `.git`, no `data/`, and no package directory. Nothing ran -- `tests.py` failed
at import, so all 37 checks were dead, not just the ones touching missing code.
It was recovered from a zip export. **The package must live in
`proteomics_revertant/`, not at the repository root**: every module uses relative
imports (`from .analysis import ...`), so a flat extraction cannot resolve them.
If you are staring at `ModuleNotFoundError` or `ImportError: attempted relative
import`, check the layout before you debug anything else. On a Windows mount, set
`git config core.fileMode false` or every file reads as modified.

## Conventions

- Comments explain *why*, especially where a simpler-looking alternative was
  rejected. Several comment blocks are load-bearing documentation of failed
  approaches; do not compress them away.
- Every statistical claim in a docstring should have a test behind it.
- p-values: all per-protein tests are two-sided. The TOST equivalence columns
  (`*_equiv_P`) and the two omnibus permutation p-values are one-sided by
  construction and correctly so. `t_two_sided` checks both cases.
- Output files are TSV. Figures are PNG at 170 dpi.
