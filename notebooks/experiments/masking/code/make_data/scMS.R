# File to generate synthetic data with scMultiSim.

library(scMultiSim)

library(dplyr)
args <- commandArgs(trailingOnly = TRUE)
cache_path = args[1]
base_grn <- read.csv(paste0(cache_path,"/", "food_for_R.csv"), header=TRUE, stringsAsFactors=FALSE)
rownames(base_grn) <- base_grn$X
base_grn$X <- NULL
#num_genes <- 110

# Count simulation
n_samples <- 8000
tree <- Phyla1()
seed <- 0
n_cifs <- 50
cif_sigma <- 0.5
diff_cif_fraction <- 0.8
intrinsic_noise <- 0.5
# Noise and Batch effect
noise <- FALSE
noise_alpha_mean <- 1e4
batch <- FALSE
n_batches <- 2
batch_effect <- 1


set.seed(seed)

save_data <- function(results, dir_name) {
  if (!dir.exists(file.path(dir_name))) {
     dir.create(file.path(dir_name))
  } 
  for (n in names(results)) {
    if (startsWith(n, ".")) {
      save_data(results[[n]], dir_name)
      next
    }
    if (class(results[[n]])[1] == "data.frame") {
      write.csv(results[[n]], file = paste0(dir_name, "/", n, ".csv"))
    } else if (class(results[[n]])[1] == "matrix") {
      write.csv(results[[n]], file = paste0(dir_name, "/", n, ".csv"))
    } else if (class(results[[n]]) == "list") {
      capture.output(summary(results[[n]]),
                     file = paste0(dir_name, "/", n, ".txt"))
    }
  }
}

options <- list(
  rand.seed = seed,
  GRN = base_grn,
  num.cells = n_samples,
  num.cifs = n_cifs,
  cif.sigma = cif_sigma,
  tree = tree,
  diff.cif.fraction = diff_cif_fraction,
  do.velocity = FALSE,
  intrinsic.noise = intrinsic_noise
  #num.genes = num_genes  
)

results <- sim_true_counts(options)

if (noise) {
  print("---Adding Noise---")
  add_expr_noise(results,
                 alpha_mean = noise_alpha_mean,
                 randseed = seed)
}

if (batch) {
  print("---Adding Batch---")
  divide_batches(results,
                 nbatch = n_batches,
                 effect = batch_effect,
                 randseed = seed)
}

save_data(results = results, dir_name = paste0(cache_path, "/","R_out"))
