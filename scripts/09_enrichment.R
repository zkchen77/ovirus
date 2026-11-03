
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
