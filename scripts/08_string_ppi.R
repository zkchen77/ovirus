
rm(list = ls())
suppressPackageStartupMessages({
  library(STRINGdb)
  library(igraph)
})

ROOT <- getwd()
IN   <- file.path(ROOT, "..", "data", "string")
OUT  <- file.path(ROOT, "..", "data", "ppi")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

SCORE_WEB <- 700     # matches the web-interface export used for Fig 2D-F
SCORE_DB  <- 400     # STRINGdb R layout threshold

patterns <- c("String_PatternI", "String_PatternII", "String_PatternIII")

for (pat in patterns) {
  tsv <- file.path(IN, paste0(pat, ".tsv"))
  if (!file.exists(tsv)) { message("missing ", tsv); next }

  message("[08] ", pat)
  string_db <- STRINGdb$new(version = "12", species = 9606,
                            score_threshold = SCORE_DB,
                            input_directory = tempdir())

  raw <- read.delim(tsv, stringsAsFactors = FALSE)
  
  # ← Web page export is on a scale of 0-1, converted to 0-1000
  raw$combined_score <- raw$combined_score * 1000   
  
  # web export columns: #node1 node1_name node2 node2_name node1_string_id ...
  # combined_score
  net <- raw[raw$combined_score >= SCORE_WEB, c("node1_string_id", "node2_string_id", "combined_score")]
  colnames(net) <- c("from", "to", "weight")
  net$weight_norm <- net$weight / 1000

  g <- graph_from_data_frame(net[, c("from", "to")], directed = FALSE)
  E(g)$weight <- net$weight_norm
  g <- simplify(g)

  message("     ", vcount(g), " nodes, ", ecount(g),
          " edges (combined_score >= ", SCORE_WEB, ")")

  # ---- topology summary
  deg <- degree(g); btw <- betweenness(g); cl <- cluster_fast_greedy(g)
  summ <- data.frame(pattern = pat,
                     nodes = vcount(g), edges = ecount(g),
                     density = round(edge_density(g), 4),
                     n_modules = length(cl),
                     top_hub = names(sort(deg, decreasing = TRUE))[1],
                     top_hub_degree = max(deg),
                     top_betweenness = names(sort(btw, decreasing = TRUE))[1])
  write.csv(summ, file.path(OUT, paste0(pat, "_topology.csv")), row.names = FALSE)

  # ---- edge list with weights (for re-plotting without STRING access)
  el <- as_data_frame(g, what = "edges")
  write.csv(el, file.path(OUT, paste0(pat, "_edges.csv")), row.names = FALSE)

  # ---- hubs table
  hubs <- data.frame(gene = V(g)$name, degree = deg, betweenness = btw)
  hubs <- hubs[order(-hubs$degree), ]
  write.csv(hubs, file.path(OUT, paste0(pat, "_hubs.csv")), row.names = FALSE)
}

message("[08] done.")
