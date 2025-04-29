"""
Test run of the bicycle model with synthetic data for bottleneck finding.

Generates diagostics and output data in the data subdirectory.



"""

import os
import time
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.tuner.tuning import Tuner


from bicycle.model import BICYCLE
from bicycle.utils.data import create_data, create_loaders
from bicycle.utils.general import get_custom_name, get_id
from bicycle.dictlogger import DictLogger
from bicycle.callbacks import (
    CustomModelCheckpoint,
    GenerateCallback,
    MyLoggerCallback,
)

SEED = 1
pl.seed_everything(SEED)
np.random.seed(SEED)

torch.set_float32_matmul_precision("high")

GPU_DEVICE = 0

# configure output paths
## subdirectories model and plots 
## in subdirectory data of the current path
OUT_PATH = Path(os.path.join(".", "notebooks/experiments/bottleneck/data"))
MODELS_PATH = Path(os.path.join(OUT_PATH, "models"))
PLOTS_PATH = Path(os.path.join(OUT_PATH, "plots"))

for directory in [MODELS_PATH, PLOTS_PATH]:
    directory.mkdir(parents=True, exist_ok=True)

# assign id to run
run_id = get_id(MODELS_PATH, prefix="test_run_")

for directory in [MODELS_PATH, PLOTS_PATH]:
    directory.joinpath(run_id).mkdir(parents=True, exist_ok=True)


# create synthetic data
## parameters for data creation with prefix data

data_n_genes = 10
data_n_samples_control = 500
data_n_samples_per_perturbation = 250
data_make_counts=True
data_train_gene_ko=np.arange(0, data_n_genes, 1).astype(str)
data_test_gene_ko = []
while len(data_test_gene_ko)<= 20:
    data_test_gene_ko.append(tuple(np.random.choice(10, size=2, replace=False).astype(str)))
    data_test_gene_ko=list(set(data_test_gene_ko))

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
# save data for later validation
np.save(os.path.join(MODELS_PATH, run_id, "check_sim_samples.npy"), samples)
np.save(os.path.join(MODELS_PATH, run_id, "check_sim_regimes.npy"), sim_regime)
np.save(os.path.join(MODELS_PATH, run_id, "check_sim_beta.npy"), beta)
np.save(os.path.join(MODELS_PATH, run_id, "check_sim_gt_interv.npy"), gt_interv)


# initialize data loaders
validation_size = 0.2
batch_size = 10_000

train_loader, validation_loader, test_loader, _ = create_loaders(samples=samples,
                          sim_regime=sim_regime,
                          validation_size=validation_size,
                          batch_size=batch_size,
                          SEED= SEED,
                          train_gene_ko=data_train_gene_ko,
                          test_gene_ko=data_test_gene_ko,
                          num_workers=1,
                          persistent_workers=False,
                          covariates=None,
)


# initialize parameters for the bicycle class with prefix model
model_lr = 1e-3
model_gt_interv = None
model_n_genes = data_n_genes
model_n_samples = data_n_samples_control + (len(data_train_gene_ko) + len(data_test_gene_ko)) * data_n_samples_per_perturbation
model_lyapunov_penalty = True
model_perfect_interventions = True
model_rank_w_cov_factor = model_n_genes
model_optimizer = "adam"
model_optimizer_kwargs = {"betas": [0.5, 0.9]}  # Faster decay for estimates of gradient and gradient squared
model_device = torch.device(f"cuda:{GPU_DEVICE}")
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
model_mask = None
model_use_encoder = False
model_gt_beta = beta
model_train_gene_ko = data_train_gene_ko
model_test_gene_ko = data_test_gene_ko
model_use_latents = True
model_covariates = None
model_n_factors = 0
model_intervention_type = "Cas9"
model_sigma_min = 1e-3
model_T = 1.0
model_learn_T = False
model_train_only_likelihood = False
model_train_only_latents = False
model_mask_genes = []

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

model.to(model_device)

# training variables

n_epochs = 10
gradient_clip_val = 1.0
plot_epoch_callback = 1

# callback variables
swa = 250
CHECKPOINTING = True
VERBOSE_CHECKPOINTING = False
SAVE_PLOT = True
OVERWRITE = True
check_val_every_n_epoch = 1
log_every_n_steps = 1



# initialize logger for the Trainer class
loggers = [DictLogger()]

# initialize training callbacks
callbacks = [
    RichProgressBar(refresh_rate=1),
    GenerateCallback(
        MODELS_PATH.joinpath(run_id, "plotname"), 
        plot_epoch_callback=plot_epoch_callback, 
        true_beta=beta.cpu().numpy()
    ),
]
if swa > 0:
                    callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))

if CHECKPOINTING:
                    Path(os.path.join(MODELS_PATH, run_id)).mkdir(parents=True, exist_ok=True)
                    callbacks.append(
                        CustomModelCheckpoint(
                            dirpath=os.path.join(MODELS_PATH, run_id),
                            filename="{epoch}",
                            save_last=True,
                            save_top_k=1,
                            verbose=VERBOSE_CHECKPOINTING,
                            monitor="valid_loss",
                            mode="min",
                            save_weights_only=True,
                            start_after=0,
                            save_on_train_epoch_end=False,
                            every_n_epochs=1,
                        )
                    )
                    callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODELS_PATH, run_id)))

# pretraining
n_epochs_pretrain_latents = 2

if model.use_latents and n_epochs_pretrain_latents > 0:

    pretrain_callbacks = [
        RichProgressBar(refresh_rate=1),
        GenerateCallback(
            str(MODELS_PATH.joinpath(id, "_pretrain")),
            plot_epoch_callback=plot_epoch_callback,
            true_beta=beta.cpu().numpy(),
        ),
    ]

    if swa > 0:
        pretrain_callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))

    pretrain_callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODELS_PATH, run_id)))

    pretrainer = pl.Trainer(
        max_epochs=n_epochs_pretrain_latents,
        accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
        logger=loggers,
        log_every_n_steps=log_every_n_steps,
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=CHECKPOINTING,
        check_val_every_n_epoch=check_val_every_n_epoch,
        devices=[GPU_DEVICE],  # if str(device).startswith("cuda") else 1,
        num_sanity_val_steps=0,
        callbacks=pretrain_callbacks,
        gradient_clip_val=gradient_clip_val,
        default_root_dir=str(MODELS_PATH),
        gradient_clip_algorithm="value",
        deterministic=False,  # "warn",
    )
    print("PRETRAINING LATENTS!")
    start_time = time.time()
    model.train_only_likelihood = True # TODO: create alternative option for pretraining
    pretrainer.fit(model, train_loader, validation_loader)
    end_time = time.time()
    model.train_only_likelihood = False
    
    pretraining_time = float(end_time - start_time)
    

# training
trainer = pl.Trainer(
                    max_epochs=n_epochs,
                    accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
                    logger=loggers,
                    log_every_n_steps=log_every_n_steps,
                    enable_model_summary=True,
                    enable_progress_bar=True,
                    enable_checkpointing=CHECKPOINTING,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    devices=[GPU_DEVICE],  # if str(device).startswith("cuda") else 1,
                    num_sanity_val_steps=0,
                    callbacks=callbacks,
                    gradient_clip_val=gradient_clip_val,
                    default_root_dir=str(MODELS_PATH),
                    gradient_clip_algorithm="value",
                    deterministic=False,  # "warn",
                )



# log the environment
current_time = time.strftime("%j%m%d%H%M%Z", time.gmtime())

pd.DataFrame(globals().items()).to_csv(os.path.join(PLOTS_PATH, run_id, "globals.csv"))