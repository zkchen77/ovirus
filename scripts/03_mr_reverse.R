# ============================================================================
# 03_mr_reverse.R   infectious disease (exposure) -> skeletal disorder (outcome)
# ============================================================================
# Identical to 02_mr_forward.R with the roles of the two trait lists swapped.
# The reverse direction is not a formality: it is the test that defines the
# ASYMMETRY criterion used to retain pairs for pattern assignment (a pair is 
# kept only if it is significant in one direction and not the other). 
# ============================================================================

rm(list = ls())
suppressPackageStartupMessages({
  library(TwoSampleMR)
  library(ieugwasr)
})

if (Sys.getenv("OPENGWAS_JWT") != "") {
  message("OPENGWAS_JWT detected - ieugwasr will use it automatically for API calls.")
  # print(ieugwasr::get_opengwas_jwt())
  # print(ieugwasr::user())
} else {
  warning("OPENGWAS_JWT not set - API calls will fail. See README.md.")
}

ROOT        <- getwd()
CFG_DIR     <- file.path(ROOT, "..", "config")
INST_DIR    <- file.path(ROOT, "..", "data", "mr_instruments")
OUT_BASE    <- file.path(ROOT, "..", "data", "pairwise_reverse")

exposures <- readLines(file.path(CFG_DIR, "infection_outcomes.txt"))   # 39
outcomes  <- readLines(file.path(CFG_DIR, "skeletal_exposures.txt"))   # 8

dir.create(OUT_BASE, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(ROOT, "..", "data", "analysis_log_reverse.txt")

run_direction <- function(exposures, outcomes, out_base, log_file, tag) {
  
  for (outcome in outcomes) {
    dir_path <- file.path(out_base, outcome)
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
    
    for (exposure in exposures) {
      tryCatch({
        inst_file <- file.path(INST_DIR, paste0("exposure_", exposure, ".csv"))
        if (!file.exists(inst_file)) stop("no instrument file for ", exposure)
        
        exposure_data <- read.csv(inst_file)
        outcome_data  <- extract_outcome_data(
          snps           = exposure_data$SNP,
          outcomes       = outcome,
          proxies        = TRUE,          # allow LD proxies (r2 > 0.8, EUR)
          rsq            = 0.8,
          align_alleles  = 1
        )
        if (is.null(outcome_data) || nrow(outcome_data) == 0)
          stop("no overlapping variant in outcome")
        
        dat <- harmonise_data(exposure_data, outcome_data, action = 2)
        
        write.csv(dat, file.path(dir_path, paste0("outcome_snp_", exposure, ".csv")),
                  row.names = FALSE)
        
        res <- mr(dat, method_list = c("mr_egger_regression",
                                       "mr_weighted_median",
                                       "mr_ivw",
                                       "mr_simple_mode",
                                       "mr_weighted_mode"))
        write.csv(res, file.path(dir_path, paste0("outcome_res_", exposure, ".csv")),
                  row.names = FALSE)
        
        het <- mr_heterogeneity(dat, method_list = c("mr_egger_regression", "mr_ivw"))
        write.csv(het, file.path(dir_path, paste0("outcome_heterogeneity_", exposure, ".csv")),
                  row.names = FALSE)
        
        ple <- mr_pleiotropy_test(dat)
        write.csv(ple, file.path(dir_path, paste0("outcome_pleiotropy_", exposure, ".csv")),
                  row.names = FALSE)
        
        # Steiger directionality (TwoSampleMR native implementation)
        st <- try(directionality_test(dat), silent = TRUE)
        if (!inherits(st, "try-error") && nrow(st)) {
          write.csv(st, file.path(dir_path, paste0("outcome_steiger_", exposure, ".csv")),
                    row.names = FALSE)
        }
        
        msg <- sprintf("[%s] %s -> %s  nsnp=%d  IVW p=%.4g",
                       tag, exposure, outcome,
                       sum(dat$mr_keep),
                       res$pval[res$method == "Inverse variance weighted"])
        message(msg); cat(msg, "\n", file = log_file, append = TRUE)
        
      }, error = function(e) {
        msg <- sprintf("[%s] %s -> %s FAILED: %s", tag, exposure, outcome, conditionMessage(e))
        message(msg); cat(msg, "\n", file = log_file, append = TRUE)
      })
    }
  }
}

run_direction(exposures, outcomes, OUT_BASE, log_file, tag = "reverse")

message("[03] done. ", length(exposures) * length(outcomes), " pairs attempted.")
