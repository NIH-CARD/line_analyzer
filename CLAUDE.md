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
python -m proteomics_revertant.tests                               # 34 checks, ~4 min
```

`tests.py` is a self-contained runner, not pytest. Every check computes an
independent reference -- brute force, Monte Carlo, or a closed form derived a
different way -- rather than comparing the code to itself. Keep it that way.

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

## Known honest limitations -- these are findings, not bugs

- Raw dose p-values run ~0.077 against a nominal 0.05 under the null. Heavy-tailed
  clone artefacts meeting 3 residual df. Does **not** propagate to reported calls:
  0 false discoveries at FDR<0.05 across null datasets.
- The global permutation p-value has a floor of ~0.071 on a 5-clone panel. When
  the summary reports P at the floor it means the effect is as extreme as the
  design can register, not that it failed.
- The simulation plants a one-directional +0.42 CNV on 90 proteins in the
  highest-leverage clone. Under the null this makes 68% of nominal hits go up.
  That is deliberate; `_qc_directional_balance.tsv` and `_qc_coherent_blocks.tsv`
  are the detectors and both fire correctly.

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

## Fast start

```bash
ruff check .                       # 30 cosmetic findings, 18 auto-fixable
ruff check . --fix                 # start here; it is a safe first commit
python -m proteomics_revertant.tests   # must stay 34/34
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

## Conventions

- Comments explain *why*, especially where a simpler-looking alternative was
  rejected. Several comment blocks are load-bearing documentation of failed
  approaches; do not compress them away.
- Every statistical claim in a docstring should have a test behind it.
- p-values: all per-protein tests are two-sided. The TOST equivalence columns
  (`*_equiv_P`) and the two omnibus permutation p-values are one-sided by
  construction and correctly so. `t_two_sided` checks both cases.
- Output files are TSV. Figures are PNG at 170 dpi.
