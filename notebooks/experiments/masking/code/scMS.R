# File to generate synthetic data with scMultiSim.

library(scMultiSim)

library(dplyr)

library(argparser)
p <- arg_parser(description = "Script for generating perurbed multi-ome scData using scMultiSim")
p <- add_argument(p, "--cache", "path to cache directory for exchanging input grn and intervention and output data.", default = ".", short = "-c")
p <- add_argument(p, "--n_samples", "number of generated samples", default = 250, short = "-s")
p <- add_argument(p, "--n_genes", "number of generated samples", default = 10, short = "-g")
p <- add_argument(p, "--noise", "if noise should be added", default = FALSE, short = "-n")
p <- add_argument(p, "--batch", "if batch effects should be added", default = FALSE, short = "-b")
p <- add_argument(p, "--intervention", "if cifs/givs should be perturbed. Dataframes to be added should be in cache path.", default = FALSE, short = "-i")
argv <- parse_args(p)
cache_path <- argv$cache
intervention <- argv$intervention
n_samples <- argv$n_samples
n_genes <- argv$n_genes
batch <- argv$batch
noise <- argv$noise
base_grn <- read.csv(paste0(cache_path,"/", "food_for_R.csv"), header=TRUE, stringsAsFactors=FALSE)
rownames(base_grn) <- base_grn$X
base_grn$X <- NULL

verbose <- TRUE
manual_pert <- FALSE
# Count simulation
#unregulated_gene_ratio <- 0
tree <- Phyla1()
seed <- 0
n_cifs <- 50
cif_sigma <- 1
diff_cif_fraction <- 0.8
intrinsic_noise <- 0.5
atac_effect <- 0.8
# region_distrib <- c(0.1, 0.5, 0.4)
# Noise and Batch effect
noise_alpha_mean <- 1e4
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

check_file <- function(file, n1, n2) {
  if(file.exists(file)){
    print(paste0(file, " exists"))
    df <- read.csv(paste0(file), header=FALSE)
    mtx <- as.matrix(df)
    return(mtx)
  } else {
    return(matrix(1, n1, n2))
  }
}

mod_cif_giv <- function(i, cif, giv, meta) {
  
  if (i == 1){

    print("modifiying kon")

    mod_giv <- check_file(paste0(cache_path,"/", "giv_kon_mod.csv"), nrow(giv), ncol(giv))
    mod_cif <- check_file(paste0(cache_path,"/", "cif_kon_mod.csv"), nrow(cif), ncol(cif))

    cif <- cif*mod_cif
    giv <- giv*mod_giv

    if (verbose){
      write.csv(cif, paste0(cache_path,"/", "cif_kon.csv"))
      write.csv(giv, paste0(cache_path,"/", "giv_kon.csv"))
    }

    if (manual_pert){
      print("Perturbing cif kon manually...")
      n_pert_per_tf <- as.integer(n_samples / n_genes)
      if (verbose){
        print(paste0("nrow(cif): ", nrow(cif)))
        print(paste0("n_pert_per_tf: ", n_pert_per_tf))
      }
      for (n in 1:n_genes) {
        start_index <- (n-1) * n_pert_per_tf + 1
        if (verbose){
          print(paste0("start_index: ", start_index))
        }
        for (i in start_index:(n_pert_per_tf + start_index - 1)) {
          cif[i, (n_cifs + n)] <- min(cif)
        }
      }
    }

    return(list(cif, giv))
  }

  if (i == 2){
    print("modifiying koff")
    mod_giv <- check_file(paste0(cache_path,"/", "giv_koff_mod.csv"), nrow(giv), ncol(giv))
    mod_cif <- check_file(paste0(cache_path,"/", "cif_koff_mod.csv"), nrow(cif), ncol(cif))

    cif <- cif*mod_cif
    giv <- giv*mod_giv

    if (verbose){
      write.csv(cif, paste0(cache_path,"/", "cif_koff.csv"))
      write.csv(giv, paste0(cache_path,"/", "giv_koff.csv"))
    }
    if (manual_pert){
      print("Perturbing cif koff manually...")
      n_pert_per_tf <- as.integer(n_samples / n_genes)
      if (verbose){
        print(paste0("nrow(cif): ", nrow(cif)))
        print(paste0("n_pert_per_tf: ", n_pert_per_tf))
      }
      for (n in 1:n_genes) {
        start_index <- (n-1) * n_pert_per_tf + 1
        if (verbose){
          print(paste0("start_index: ", start_index))
        }
        for (i in start_index:(n_pert_per_tf + start_index - 1)) {
          cif[i, (n_cifs + n)] <- max(cif)
        }
      }
    }
    return(list(cif, giv))

  }

  if (i == 3){
    print("not modifiying s")
    mod_giv <- check_file(paste0(cache_path,"/", "giv_s_mod.csv"), nrow(giv), ncol(giv))
    mod_cif <- check_file(paste0(cache_path,"/", "cif_s_mod.csv"), nrow(cif), ncol(cif))

    cif <- cif*mod_cif
    giv<- giv*mod_giv

    if (verbose){
      write.csv(cif, paste0(cache_path,"/", "cif_s.csv"))
      write.csv(giv, paste0(cache_path,"/", "giv_s.csv"))
    }
    if (manual_pert){
      print("Perturbing cif s manually...")
      n_pert_per_tf <- as.integer(n_samples / n_genes)
      if (verbose){
        print(paste0("nrow(cif): ", nrow(cif)))
        print(paste0("n_pert_per_tf: ", n_pert_per_tf))
      }
      for (n in 1:n_genes) {
        start_index <- (n-1) * n_pert_per_tf + 1
        if (verbose){
          print(paste0("start_index: ", start_index))
        }
        for (i in start_index:(n_pert_per_tf + start_index - 1)) {
          cif[i, n] <- min(cif)
        }
      }
    }
    return(list(cif, giv))
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
  intrinsic.noise = intrinsic_noise,
  #unregulated.gene.ratio = unregulated_gene_ratio,
  num.genes = n_genes+1,
  mod.cif.giv = if (intervention) mod_cif_giv else NULL,
  atac.effect = atac_effect
  # region.distrib = region_distrib
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
