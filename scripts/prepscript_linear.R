# ---------------------------------------------------------------------------
# Prep the full iNDI bulk RNA-seq AnnData for the line_analyzer pipeline.
#
# Unlike the original script, this does NOT subset to one gene/variant. It
# builds the pipeline's required sample-level columns (clone, dose, lineage,
# state, mutation) for every gene/variant combination present, and writes ONE
# h5ad with everything in it. Splitting into one dataset per mutation happens
# downstream in Python (proteomics_revertant.io_adata --split-cols), so you
# do this R step exactly once.
#
# ASSUMPTIONS -- check these against your actual data before trusting output:
#   1. `zygosity` values are exactly wt / het / hom / rev. If there are other
#      values (e.g. "compound_het"), the dose map below will error loudly
#      rather than silently mis-assigning a dose -- that's intentional.
#   2. `jax_id` is NOT assumed unique per genotype (real data showed it isn't
#      -- see the composite `clone` key below, which handles this instead).
#   3. `lineage` reflects a known derivation tree: parental is its own
#      lineage; hom-edit clones are an independent editing event from
#      parental (own lineage); het-edit clones are also directly from
#      parental, but their revertant is a second editing step performed ON
#      the het clone, so het and its revertant share a lineage. If any
#      mutation in your data was made differently (e.g. hom derived from
#      het, or multiple independent het electroporations), fix the
#      case_when below for that mutation specifically.
#   4. Adjust `qc_drop_col` / `qc_drop_value` to whatever your real QC-exclude
#      column is (the example script used `differentiation_person ==
#      "automated"`; that column did not appear in the sampleData sample you
#      pasted, so confirm the real name before running).
#   5. `GRN1` is normalised to `GRN` (assumed typo -- confirm this is right).
#   6. Compound (two-edit) variants like "R317W_R406W" are EXCLUDED from the
#      per-mutation analysis -- a single dose axis cannot represent them.
#      They print to the console when dropped; if you want to analyse them,
#      it needs to be a bespoke comparison outside this pipeline's dose model.
# ---------------------------------------------------------------------------

library(zellkonverter)
library(SingleCellExperiment)
library(dplyr)

sce <- readH5AD(
  "/data/CARD_AUX/users/paquolaac/projects/iNDI_bulk_RNAseq/analysis/anndata/qc/filter/_o/adata.h5ad"
)
sce
assayNames(sce)
colnames(colData(sce))
dim(sce)

counts_mat <- assay(sce, "counts")
rownames(counts_mat) <- gsub("\\..*", "", rownames(counts_mat))

sampleData <- as.data.frame(colData(sce))
rownames(sampleData) <- colnames(counts_mat)
stopifnot(identical(colnames(counts_mat), rownames(sampleData)))

# ---- fill WT for gene / variant, as before ---------------------------------
sampleData$gene <- as.character(sampleData$gene)
sampleData$gene[trimws(sampleData$gene) == "" | is.na(sampleData$gene)] <- "WT"

sampleData$variant <- as.character(sampleData$variant)
sampleData$variant[trimws(sampleData$variant) == "" | is.na(sampleData$variant)] <- "WT"

# ---- optional QC exclusion --------------------------------------------------
# Keep only automated differentiation samples. Manual samples are marked with
# the differentiator's initials (heterogeneous, not one fixed value), so this
# KEEPS rows matching qc_keep_value rather than dropping a matched value.
# Set qc_drop_col <- NULL to skip this filter entirely.
qc_drop_col   <- "differentiation_person"
qc_keep_value <- "automated"
if (!is.null(qc_drop_col) && qc_drop_col %in% names(sampleData)) {
  keep <- sampleData[[qc_drop_col]] %in% qc_keep_value
  cat(sprintf("\nKeeping %d/%d samples where %s == %s (dropping manual/initials rows)\n",
              sum(keep), length(keep), qc_drop_col, qc_keep_value))
  sampleData    <- sampleData[keep, , drop = FALSE]
  counts_mat    <- counts_mat[, rownames(sampleData), drop = FALSE]
} else if (!is.null(qc_drop_col)) {
  warning(sprintf(
    "qc_drop_col %s not found in sampleData -- no QC rows dropped. Check the real column name.",
    qc_drop_col))
}

# ---- required pipeline columns ---------------------------------------------

# GRN1 vs GRN, same jax_id/variant/zygosity: treated as a naming typo, not a
# real distinct gene. CONFIRM this against your metadata before trusting it --
# if GRN1 is a real distinct locus, remove this line.
sampleData$gene[sampleData$gene == "GRN1"] <- "GRN"

# FUS should only have one variant, R216C -- R126C was a mislabel, not a
# second reversion target (confirmed).
mislabelled <- sampleData$gene == "FUS" & sampleData$variant == "R126C"
if (any(mislabelled)) {
  cat(sprintf("\nRelabelling %d FUS/R126C rows to FUS/R216C (confirmed mislabel)\n",
              sum(mislabelled)))
  sampleData$variant[mislabelled] <- "R216C"
}

# Compound/double-edited lines (e.g. variant "R317W_R406W") carry two edits on
# one clone. This pipeline's single ordinal dose axis cannot represent that
# (README S8: "one edit per line") -- excluding rather than mis-modelling.
# Detects any non-WT variant string containing an underscore-joined second
# mutation code; adjust the pattern if your naming convention differs.
compound <- sampleData$gene != "WT" & grepl("_", sampleData$variant)
if (any(compound)) {
  cat(sprintf("\nExcluding %d libraries with compound (two-edit) variants, not supported by a single dose axis: %s\n",
              sum(compound), paste(unique(sampleData$variant[compound]), collapse = ", ")))
  sampleData <- sampleData[!compound, , drop = FALSE]
  counts_mat <- counts_mat[, rownames(sampleData), drop = FALSE]
}

# clone: jax_id is NOT reliably one-clone-per-genotype in this dataset -- some
# jax_ids are reused across different zygosity states or different variants
# (e.g. a line resampled after further editing, or two reversions sharing an
# id). Key on the full genotype so "clone" always means "one genotype, one
# biological replicate unit", which is what the pipeline actually requires.
# If jax_id was already unique per genotype this is a no-op; where it wasn't,
# this is what forces the correct separation.
sampleData$clone <- paste(sampleData$jax_id, sampleData$gene, sampleData$variant,
                          sampleData$zygosity, sep = "_")

# dose: allele count of the mutant variant, derived from zygosity.
# IMPORTANT: zygosity comes back from readH5AD as a factor (pandas categorical
# -> R factor). Indexing a named vector with a factor uses the factor's
# alphabetical integer codes as POSITIONS, not its labels -- as.character()
# first is required or every dose is silently wrong.
dose_map <- c(wt = 0L, het = 1L, hom = 2L, rev = 0L)
zyg_chr <- as.character(sampleData$zygosity)
if (!all(unique(zyg_chr) %in% names(dose_map))) {
  bad <- setdiff(unique(zyg_chr), names(dose_map))
  stop("Unrecognised zygosity values, extend dose_map before proceeding: ",
       paste(bad, collapse = ", "))
}
sampleData$dose <- unname(dose_map[zyg_chr])

# state: a readable genotype label
sampleData$state <- ifelse(
  sampleData$gene == "WT", "wild_type",
  paste(sampleData$zygosity, "edit", sep = "_")
)
sampleData$state[sampleData$zygosity == "rev"] <- paste0(sampleData$gene[sampleData$zygosity == "rev"], "_revertant")

# mutation: the grouping key each pipeline dataset will be split on downstream
sampleData$mutation <- ifelse(
  sampleData$gene == "WT", "WT",
  paste(sampleData$gene, sampleData$variant, sep = "_")
)

# lineage: reflects the real derivation tree, not just gene_variant --
#   parental (wt/wt)
#     -> hom (mut/mut): one independent editing step from parental
#     -> het (mut/wt):  one editing step from parental
#          -> revertant (wt/rev): a SECOND editing step, from the het clone
#   So het and its revertant share a lineage (same off-target/culture history
#   minus the causal allele); hom is its own lineage; parental is its own.
sampleData$lineage <- dplyr::case_when(
  sampleData$gene == "WT"       ~ "parental",
  sampleData$zygosity == "hom"  ~ paste(sampleData$gene, sampleData$variant, "hom_edit", sep = "_"),
  sampleData$zygosity %in% c("het", "rev") ~
    paste(sampleData$gene, sampleData$variant, "het_edit", sep = "_"),
  TRUE ~ NA_character_
)
if (any(is.na(sampleData$lineage))) {
  stop("Unhandled zygosity value(s) when assigning lineage -- extend the case_when above: ",
       paste(unique(sampleData$zygosity[is.na(sampleData$lineage)]), collapse = ", "))
}

# tech_rep: replicate index within a clone
sampleData <- sampleData %>%
  group_by(clone) %>%
  mutate(tech_rep = row_number()) %>%
  ungroup() %>%
  as.data.frame()
rownames(sampleData) <- colnames(counts_mat)  # dplyr drops rownames -- restore

# plex / batch: use plate_name if present, else a single batch
sampleData$plex <- if ("plate_name" %in% names(sampleData)) {
  as.character(sampleData$plate_name)
} else {
  "1"
}

stopifnot(identical(colnames(counts_mat), rownames(sampleData)))

# sanity check: within a clone, state/dose/lineage must agree (this is what
# the pipeline itself will enforce on the Python side -- fail here instead)
inconsistent <- sampleData %>%
  group_by(clone) %>%
  summarise(n_state = n_distinct(state), n_dose = n_distinct(dose),
            n_lineage = n_distinct(lineage), .groups = "drop") %>%
  filter(n_state > 1 | n_dose > 1 | n_lineage > 1)
if (nrow(inconsistent) > 0) {
  cat("\n--- inconsistent clones: full offending rows ---\n")
  print(
    sampleData %>%
      filter(clone %in% inconsistent$clone) %>%
      select(clone, any_of("sample"), gene, variant, zygosity, dose, state, lineage) %>%
      arrange(clone) %>%
      as.data.frame()
  )
  stop("Some clones have inconsistent state/dose/lineage across replicate rows -- see rows above, fix before exporting.")
}

cat(sprintf("\n%d samples, %d clones, %d mutation groups (incl. WT)\n\n",
            nrow(sampleData), n_distinct(sampleData$clone), n_distinct(sampleData$mutation)))
print(table(sampleData$mutation, sampleData$zygosity))

# ---- assemble and write a single h5ad --------------------------------------
sce_out <- SingleCellExperiment(
  assays = list(counts = counts_mat),
  colData = sampleData
)

writeH5AD(sce_out, file = "adata_prepped_for_pipeline.h5ad", X_name = "counts")
cat("\nwrote adata_prepped_for_pipeline.h5ad\n")
