"""
Test run of the bicycle model with synthetic data for bottleneck finding.

Generates diagostics and output data in the data subdirectory.

TODO: set data_device to cpu again

"""
import time

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging, DeviceStatsMonitor
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.loggers import Logger


from bicycle.model import BICYCLE
from bicycle.utils.data import create_data, create_loaders, get_diagonal_mask
from bicycle.utils.general import get_id
from bicycle.utils.plotting import plot_training_results
from bicycle.callbacks import (
    CustomModelCheckpoint,
    GenerateCallback,
    MyLoggerCallback,
)

# arguments: 
SEED = 1
scale_factor = 1
GPU_DEVICES = [1]
monitor_stats = False
profile = False
CHECKPOINTING = False
progressbar_rate = 0
compile = False
compiler_kwargs = {}
compiler_fullgraph = False
compiler_dynamic = False
compiler_mode = None
loader_workers=1
trainer_precision=32
matmul_precision="high"

pl.seed_everything(SEED)
torch.set_float32_matmul_precision(matmul_precision)

"""
commonly used parameters:
n_genes -> must be >2 => scale_factor>5
n_samples_control
n_samples_per_perturbation
batch_size
n_epochs
n_epochs_pretrain_latents
are default divided by 'scale_factor'
"""

# configure output paths
## subdirectories model and plots
## in subdirectory data of the current path
print("Setting output paths...")

OUT_PATH = Path(os.path.join(".", "notebooks/experiments/bottleneck/data"))
MODELS_PATH = OUT_PATH.joinpath("models")
PLOTS_PATH = OUT_PATH.joinpath("plots")

print("Creating output directories...")
for directory in [MODELS_PATH, PLOTS_PATH]:
    directory.mkdir(parents=True, exist_ok=True)

# assign id to run
run_id = get_id(MODELS_PATH, prefix="test_run_")

for directory in [MODELS_PATH, PLOTS_PATH]:
    directory.joinpath(run_id).mkdir(parents=True, exist_ok=True)

MODELS_PATH = OUT_PATH.joinpath("models", run_id)
PLOTS_PATH = OUT_PATH.joinpath("plots", run_id)
print(f"Output paths are: {str(MODELS_PATH)} and {str(PLOTS_PATH)}!")

# create synthetic data
## parameters for data creation with prefix data
print("Setting data creation parameters...")
data_n_genes = 10 # needs to be a round number
data_n_samples_control = 500
data_n_samples_per_perturbation = 100
data_make_counts=True
data_train_gene_ko=list(np.arange(0, data_n_genes, 1).astype(str))
data_test_gene_ko = []
while len(data_test_gene_ko)<= data_n_genes*2:
    data_test_gene_ko.append(tuple(np.random.choice(data_n_genes, size=2, replace=False).astype(str)))
    data_test_gene_ko=list(set(data_test_gene_ko))
data_test_gene_ko = [f"{x[0]},{x[1]}" for x in data_test_gene_ko]

data_graph_type="erdos-renyi"
data_edge_assignment="random-uniform"
data_sem="linear-ou"
data_make_contractive=True
data_verbose=False
data_device=torch.device("cpu")
data_intervention_type="Cas9"
data_T=1.0
data_library_size_range=[10 * data_n_genes, 100 * data_n_genes]
data_graph_kwargs = {
    "abs_weight_low": 0.25,
    "abs_weight_high": 0.95,
    "p_success": 0.5,
    "expected_density": 2,
    "noise_scale": 0.5,
    "intervention_scale": 0.1,
}
print("Creating synthetic data...")
gt_dyn, intervened_variables, samples, gt_interv, sim_regime, beta = create_data(
    n_genes = data_n_genes,
    n_samples_control = data_n_samples_control,
    n_samples_per_perturbation = data_n_samples_per_perturbation,
    make_counts = data_make_counts,
    train_gene_ko = data_train_gene_ko,
    test_gene_ko = data_test_gene_ko,
    graph_type = data_graph_type,
    edge_assignment = data_edge_assignment,
    sem = data_sem,
    make_contractive = data_make_contractive,
    verbose = data_verbose,
    device = data_device,
    intervention_type = data_intervention_type,
    T = data_T,
    library_size_range = data_library_size_range,
    **data_graph_kwargs,
)
# initialize data loaders
print("Initializing Dataloaders...")
validation_size = 0.2
batch_size = 1000

train_loader, validation_loader, test_loader = create_loaders(samples=samples,
                          sim_regime=sim_regime,
                          validation_size=validation_size,
                          batch_size=batch_size,
                          SEED= SEED,
                          train_gene_ko=data_train_gene_ko,
                          test_gene_ko=data_test_gene_ko,
                          persistent_workers=False,
                          covariates=None,
                          num_workers= loader_workers

)

# initialize parameters for the bicycle class with prefix model
print("Initializing model parameters...")
model_lr = 1e-3
model_gt_interv = gt_interv
model_n_genes = samples.shape[1]
model_n_samples = samples.shape[0]
model_lyapunov_penalty = True
model_perfect_interventions = True
model_rank_w_cov_factor = model_n_genes
model_optimizer = "adam"
# Faster decay for estimates of gradient and gradient squared:
model_optimizer_kwargs = {"betas": [0.5, 0.9]}
model_device = torch.device(f"cuda:{GPU_DEVICES[0]}")
model_scale_l1 = 0.1
model_scale_spectral = 1.0
model_scale_lyapunov = 1.0
model_scale_kl = 1.0
model_early_stopping = False
model_early_stopping_min_delta = 0.01
model_early_stopping_patience = 500
model_early_stopping_p_mode = True
model_x_distribution = "Multinomial"
model_x_distribution_kwargs = {}
model_init_tensors = None
model_mask = get_diagonal_mask(n_genes=model_n_genes, device=model_device)
model_use_encoder = False
model_gt_beta = beta
model_train_gene_ko = data_train_gene_ko
model_test_gene_ko = data_test_gene_ko
model_use_latents = True    # Should be the same as data_make_counts
model_covariates = None
model_n_factors = 0
model_intervention_type = "Cas9"
model_sigma_min = 1e-3
model_T = 1.0
model_learn_T = False
model_train_only_likelihood = False
model_train_only_latents = False
model_mask_genes = []

print("Initializing BICYCLE model...")
model = BICYCLE(
    lr = model_lr,
    gt_interv = model_gt_interv,
    n_genes = model_n_genes,
    n_samples = model_n_samples,
    lyapunov_penalty = model_lyapunov_penalty,
    perfect_interventions = model_perfect_interventions,
    rank_w_cov_factor = model_rank_w_cov_factor,
    optimizer = model_optimizer,
    optimizer_kwargs = model_optimizer_kwargs,
    device = model_device,
    scale_l1 = model_scale_l1,
    scale_spectral = model_scale_spectral,
    scale_lyapunov = model_scale_lyapunov,
    scale_kl = model_scale_kl,
    early_stopping = model_early_stopping,
    early_stopping_min_delta = model_early_stopping_min_delta,
    early_stopping_patience = model_early_stopping_patience,
    early_stopping_p_mode = model_early_stopping_p_mode,
    x_distribution = model_x_distribution,
    x_distribution_kwargs = model_x_distribution_kwargs,
    init_tensors = model_init_tensors,
    mask = model_mask,
    use_encoder = model_use_encoder,
    gt_beta = model_gt_beta,
    train_gene_ko = model_train_gene_ko,
    test_gene_ko = model_test_gene_ko,
    use_latents = model_use_latents,
    covariates = model_covariates,
    n_factors = model_n_factors,
    intervention_type = model_intervention_type,
    sigma_min = model_sigma_min,
    T = model_T,
    learn_T = model_learn_T,
    train_only_likelihood = model_train_only_likelihood,
    train_only_latents = model_train_only_latents,
    mask_genes = model_mask_genes,
)

if compile:
    torch.compile(model=model, 
        fullgraph = compiler_fullgraph,
        dynamic = compiler_dynamic,
        mode=compiler_mode)
model.to(model_device)


# training variables
n_epochs = 1000
gradient_clip_val = 1.0

print("Training data:")
print(f"- Number of training samples: {len(train_loader.dataset)}")
if validation_size > 0:
    print(f"- Number of validation samples: {len(validation_loader.dataset)}")

# callback variables
swa = 250
VERBOSE_CHECKPOINTING = False
check_val_every_n_epoch = 1
log_every_n_steps = 1

# pretraining
n_epochs_pretrain_latents = 100

if model.use_latents and n_epochs_pretrain_latents > 0:
    print(f"Pretraining model latents for {n_epochs_pretrain_latents} epochs...")
    
    pretrainer = pl.Trainer(
        max_epochs=n_epochs_pretrain_latents,
        accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
        log_every_n_steps=log_every_n_steps,
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=CHECKPOINTING,
        check_val_every_n_epoch=check_val_every_n_epoch,
        devices=GPU_DEVICES,  # if str(device).startswith("cuda") else 1,
        num_sanity_val_steps=0,
        gradient_clip_val=gradient_clip_val,
        default_root_dir=str(MODELS_PATH),
        gradient_clip_algorithm="value",
        deterministic=False,  # "warn",
        precision=trainer_precision,
    )
    print("PRETRAINING LATENTS!")
    start_time = time.time()
    model.train_only_likelihood = True # TODO: create alternative option for pretraining
    pretrainer.fit(model, train_loader, validation_loader)
    end_time = time.time()
    model.train_only_likelihood = False
    pretraining_time = float(end_time - start_time)
    print(f"Pretraining took {pretraining_time} seconds.")

print(f"Training the model for {n_epochs} epochs...")
# training
trainer = pl.Trainer(
    max_epochs=n_epochs,
    accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
    log_every_n_steps=log_every_n_steps,
    enable_model_summary=True,
    enable_progress_bar=True,
    enable_checkpointing=CHECKPOINTING,
    check_val_every_n_epoch=check_val_every_n_epoch,
    devices=GPU_DEVICES,  # if str(device).startswith("cuda") else 1,
    num_sanity_val_steps=0,
    gradient_clip_val=gradient_clip_val,
    default_root_dir=str(MODELS_PATH),
    gradient_clip_algorithm="value",
    deterministic=False,  # "warn",
    precision=trainer_precision,
)

print("TRAINING THE MODEL!")

start_time = time.time()
trainer.fit(model, train_loader, validation_loader)
end_time = time.time()

training_time = float(end_time - start_time)
print(f"Training took {training_time} seconds.")


# log the environment
pd.DataFrame(globals().items()).to_csv(os.path.join(PLOTS_PATH, "globals.csv"))
