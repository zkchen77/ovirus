# ============================================================================
# 01_mr_extract_instruments.R
# ----------------------------------------------------------------------------
# Instrument selection for every trait used in the study (8 skeletal disorders
# + 39 infectious diseases = 47 GWAS IDs).
#
#       F = (beta.exposure / se.exposure)^2
#
# LD CLUMPING (explicit, previously undocumented)
# -----------------------------------------------
# we now state explicitly rather than rely on:
#     clump = TRUE,  r2 = 0.001,  kb = 10000
# Reference panel: 1000 Genomes EUR (via the IEU OpenGWAS LD reference API).
#
# REQUIREMENT: an OpenGWAS API token.  See README.md section
# "OpenGWAS API access".
# ============================================================================

rm(list = ls())
suppressPackageStartupMessages({
  library(TwoSampleMR)
  library(ieugwasr)
})

# --- OpenGWAS authentication -------------------------------------------------
# Generate a JWT at https://api.opengwas.io/profile/ and store it in ~/.Renviron as
#   OPENGWAS_JWT=xxxxxxxxxxxx

  if (Sys.getenv("OPENGWAS_JWT") != "") {
  message("OPENGWAS_JWT detected - ieugwasr will use it automatically for API calls.")
  # print(ieugwasr::get_opengwas_jwt())
  # print(ieugwasr::user())
} else {
  warning("OPENGWAS_JWT not set - API calls will fail. See README.md.")
}

# --- paths
ROOT        <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."), ".."))
ROOT        <- getwd()
CFG_DIR     <- file.path(ROOT, "..", "config")
INST_DIR    <- file.path(ROOT, "..", "data", "mr_instruments")
OUT_DIR     <- file.path(ROOT, "..", "data", "mr_qc")
dir.create(INST_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_DIR,  recursive = TRUE, showWarnings = FALSE)

# --- instrument selection parameters (single source of truth)
P_THRESHOLD   <- 5e-6     # relaxed from 5e-8; justified above
CLUMP         <- TRUE
CLUMP_R2      <- 0.001
CLUMP_KB      <- 10000
P2_THRESHOLD  <- 5e-6     # secondary-screen threshold (outcome side), unused here

exposure_path <- file.path(CFG_DIR, "all_traits.txt")
stopifnot(file.exists(exposure_path))
trait_list <- readLines(exposure_path)

audit <- data.frame()

for (elem in trait_list) {
  tryCatch({
    message("[01] ", elem)

    dat <- extract_instruments(
      outcomes      = elem,
      p1            = P_THRESHOLD,
      clump         = CLUMP,
      r2            = CLUMP_R2,
      kb            = CLUMP_KB
    )

    if (is.null(dat) || nrow(dat) == 0) {
      message("     no instrument at p < ", P_THRESHOLD)
      next
    }

    f <- file.path(INST_DIR, paste0("exposure_", elem, ".csv"))
    write.csv(dat, file = f, row.names = FALSE)

    # ---- threshold audit
    audit <- rbind(audit, data.frame(
      id            = elem,
      n_p5e6        = sum(dat$pval.exposure < 5e-6),
      n_p1e6        = sum(dat$pval.exposure < 1e-6),
      n_p5e7        = sum(dat$pval.exposure < 5e-7),
      n_p5e8        = sum(dat$pval.exposure < 5e-8),
      F_min         = min((dat$beta.exposure / dat$se.exposure)^2),
      F_mean        = mean((dat$beta.exposure / dat$se.exposure)^2),
      N             = suppressWarnings(mean(dat$samplesize.exposure, na.rm = TRUE))
    ))

    message("     ", nrow(dat), " instruments written -> ", basename(f))
  }, error = function(e) {
    message("     FAILED: ", conditionMessage(e))
  })
}

if (nrow(audit)) {
  write.csv(audit, file.path(OUT_DIR, "QC0_threshold_audit.csv"), row.names = FALSE)
  message("[01] threshold audit written: ",
          sum(audit$n_p5e8 == 0), "/", nrow(audit),
          " traits have NO instrument at p<5e-8")
}

message("[01] done.")
