# ============================================================================
# 09_enrichment.R
# ----------------------------------------------------------------------------
# GO-BP / KEGG / Reactome enrichment for the co-occurrence gene set of each
# infection-susceptibility pattern (Pattern I / II / III).
#
# PARAMETERS - all previously implicit, now declared here and used verbatim:
#
#   clusterProfiler::enrichGO(
#       OrgDb         = org.Hs.eg.db,
#       keyType       = "ENTREZID",
#       ont           = "BP",
#       pvalueCutoff  = 0.05,      # BH-adjusted, see below
#       qvalueCutoff  = 0.2,       # DOSE q-value (Storey), secondary filter
#       pAdjustMethod = "BH",
#       readable      = TRUE)
#
#   clusterProfiler::enrichKEGG(
#       organism      = "hsa",
#       pvalueCutoff  = 0.05,
#       qvalueCutoff  = 0.2,
#       pAdjustMethod = "BH")
#
#   ReactomePA::enrichPathway(
#       organism      = "human",
#       pvalueCutoff  = 0.05,
#       qvalueCutoff  = 0.2,
#       pAdjustMethod = "BH",
#       readable      = TRUE)
#
# BACKGROUND SET
# --------------
# The default clusterProfiler background: every Entrez gene present in the
# chosen annotation database (org.Hs.eg.db / KEGG hsa / Reactome human).
# No custom background is used, because the co-occurrence gene sets were not
# selected by any technology that would make a technology-matched background
# (e.g. a microarray detection mask) appropriate.  This is stated in Methods.
#
# MULTIPLE TESTING
# ----------------
# `p.adjust` in the output is Benjamini-Hochberg FDR over all terms in the
# database that survived the size/inclusion filters - i.e. the correction the
# reviewers asked for is applied by default and is reported in every exported
# CSV column `p.adjust`.
# ============================================================================

rm(list = ls())
suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(ReactomePA)
  library(DOSE)
})

ROOT   <- getwd()
IN_DIR <- file.path(ROOT, "..", "data", "gene_sets")
OUT    <- file.path(ROOT, "..", "data", "enrichment")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

P_CUT   <- 0.05
Q_CUT   <- 0.2
ADJ_MTH <- "BH"

patterns <- c("patternI", "patternII", "patternIII")

for (pat in patterns) {
  f <- file.path(IN_DIR, paste0(pat, "_co-occurrence genes.txt"))
  if (!file.exists(f)) { message("missing ", f); next }

  sym <- readLines(f); sym <- sym[nzchar(sym)]
  message("[09] ", pat, ": ", length(sym), " gene symbols")

  # SYMBOL -> ENTREZID (drop NA, keep one-to-one)
  mp <- bitr(sym, fromType = "SYMBOL", toType = "ENTREZID",
             OrgDb = org.Hs.eg.db, drop = TRUE)
  message("     ", nrow(mp), " mapped to Entrez")
  if (nrow(mp) < 5) next

  ego <- enrichGO(gene          = mp$ENTREZID,
                  OrgDb         = org.Hs.eg.db,
                  keyType       = "ENTREZID",
                  ont           = "BP",
                  pvalueCutoff  = P_CUT,
                  qvalueCutoff  = Q_CUT,
                  pAdjustMethod = ADJ_MTH,
                  readable      = TRUE)
  eke <- enrichKEGG(gene          = mp$ENTREZID,
                    organism      = "hsa",
                    pvalueCutoff  = P_CUT,
                    qvalueCutoff  = Q_CUT,
                    pAdjustMethod = ADJ_MTH)
  ere <- enrichPathway(gene          = mp$ENTREZID,
                       organism      = "human",
                       pvalueCutoff  = P_CUT,
                       qvalueCutoff  = Q_CUT,
                       pAdjustMethod = ADJ_MTH,
                       readable      = TRUE)

  write.csv(as.data.frame(ego), file.path(OUT, paste0("01_GO-BP_", pat, ".csv")), row.names = FALSE)
  write.csv(as.data.frame(eke), file.path(OUT, paste0("02_KEGG_",   pat, ".csv")), row.names = FALSE)
  write.csv(as.data.frame(ere), file.path(OUT, paste0("03_Reactome_", pat, ".csv")), row.names = FALSE)

  message("     GO-BP ", nrow(as.data.frame(ego)),
          " | KEGG ", nrow(as.data.frame(eke)),
          " | Reactome ", nrow(as.data.frame(ere)), " terms (all BH-adjusted)")
}

message("[09] done.")
