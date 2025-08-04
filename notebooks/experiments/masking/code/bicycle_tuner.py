"""
Test run of the bicycle model with synthetic data for evaluating masking strategies.
"""
import time
current_time = time.strftime("%j%m%d%H%M%Z", time.gmtime())

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging, DeviceStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.loggers import Logger, CSVLogger
import scanpy as sc


from model_tune import BICYCLE
from bicycle.utils.data import create_data, create_loaders, get_diagonal_mask, create_data_from_grn
from bicycle.utils.general import get_id
from bicycle.utils.plotting import plot_training_results
from bicycle.callbacks import (
    CustomModelCheckpoint,
    GenerateCallback,
    MyLoggerCallback,
)
from bicycle.utils.mask_utils import format_data, get_sparsity, normalize_tensor
from bicycle.dictlogger import DictLogger
import pickle
import shutil
import gc
import ray
from ray import tune
from ray.tune.utils import wait_for_gpu
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.train.torch import TorchTrainer
from ray.tune import RunConfig, CheckpointConfig
from ray.train import ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from datetime import timedelta



DATA_PATH = Path("/data/toulouse/bicycle/notebooks/experiments/masking/data")
GPU_DEVICES = [1]
print(str(GPU_DEVICES)[1:-1])
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICES)[1:-1]
print(f"RUNNING ON CUDA: {GPU_DEVICES}")
SEED = 1
GPU_DEVICES = [0]


early_stopping = False

# masking
masking_mode = None
use_atac_mask = False
bin_prior = False
scale_mask = 1
parameter_set = "params5"
grn_noise_factor = 0.5
normalize_mask = False
# data
data_source = "create_data"
data_sem="linear"

# scMultiSim
scMultiSim_path = DATA_PATH.parent/"scMS.R"
data_id = "data_run000"

# needs to be a round number
data_n_genes = 10
# need to be greater than 1
data_n_samples_control = 500
data_n_samples_per_perturbation = 250

# training
model_use_latents = False    # determines if latent representation is learned

batch_size = 10000
n_epochs = 5000
n_epochs_pretrain_latents = 100
validation_size = 0.2

if masking_mode != "loss" and bin_prior:
    raise NotImplementedError("masking mode must be loss for bin_prior")

masking_loss = masking_mode == "loss"
use_hard_mask = masking_mode == "hard"

pl.seed_everything(SEED)
torch.set_float32_matmul_precision("high")

# configure output paths
## subdirectories model and plots
## in subdirectory data of the current path
parameters_path = DATA_PATH/"parameters" / (parameter_set + ".pickle")


print("Setting output paths...")

OUT_PATH = DATA_PATH/"model_runs"
MODELS_PATH = OUT_PATH/"models"

print("Creating output directories...")
if not MODELS_PATH.exists():
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

# assign id to run
run_id = get_id(MODELS_PATH, prefix="tuner_run_", id_length=3)

if not MODELS_PATH.joinpath(run_id).exists():
    MODELS_PATH.joinpath(run_id).mkdir(parents=True, exist_ok=True)

MODELS_PATH = OUT_PATH.joinpath("models", run_id)
print(f"Output path is: {str(MODELS_PATH)}!")

# add a copy of the protocoll to the dest path
shutil.copy(__file__, MODELS_PATH)



n_workers = 5
data_device=torch.device("cpu")
data_train_gene_ko=[str(s) for s in range(data_n_genes)]
data_test_gene_ko = []
while len(data_test_gene_ko)< data_n_genes*2:
    data_test_gene_ko.append(tuple(np.random.choice(data_n_genes, size=2, replace=False).astype(str)))
    data_test_gene_ko=list(set(data_test_gene_ko))
data_test_gene_ko = [f"{x[0]},{x[1]}" for x in data_test_gene_ko]
data_intervention_type="Cas9"
data_graph_kwargs = {
    "abs_weight_low": 0.25,
    "abs_weight_high": 0.95,
    "p_success": 0.5,
    "expected_density": 2,
    "noise_scale": 0.5,
    "intervention_scale": 0.1,
}


if data_source == "create_data":
    if use_atac_mask:
        print("atac_mask not supported if data_source='create_data'")
    # create synthetic data
    ## parameters for data creation with prefix data
    print("Setting data creation parameters...")
    
    data_make_counts=True

    data_graph_type="erdos-renyi"
    data_edge_assignment="random-uniform"
    data_make_contractive=True    
    data_T=1.0
    data_library_size_range=[10 * data_n_genes, 100 * data_n_genes]
    

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
        verbose = False,
        device = data_device,
        intervention_type = data_intervention_type,
        T = data_T,
        library_size_range = data_library_size_range,
        **data_graph_kwargs,
    )
    samples = normalize_tensor(samples)
    # save data for later validation
    print(f"Saving data at {str(MODELS_PATH.joinpath('synthetic_data'))}...")
    MODELS_PATH.joinpath("synthetic_data").mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_samples.npy"), samples)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_regimes.npy"), sim_regime)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_beta.npy"), beta)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_gt_interv.npy"), gt_interv)

    # initialize data loaders
    print("Initializing Dataloaders...")
    
    train_loader, validation_loader, test_loader = create_loaders(
        samples=samples, # test_loader is None
        sim_regime=sim_regime,
        validation_size=validation_size,
        batch_size=batch_size,
        SEED= SEED,
        train_gene_ko=data_train_gene_ko,
        test_gene_ko=data_test_gene_ko,
        persistent_workers=False,
        covariates=None,
        num_workers= n_workers,

    )
    model_n_genes = samples.shape[1]
    model_n_samples = samples.shape[0]
    model_train_gene_ko = data_train_gene_ko
    model_test_gene_ko = data_test_gene_ko
    model_init_tensors = {}
    if masking_loss:
        values = beta.numpy()[beta>0]
        grn_noise_var = np.std(values)
        grn_noise_mean = np.mean(values)
        grn_noise_p = get_sparsity(beta)*grn_noise_factor
        salt = np.random.rand(*beta.shape) < grn_noise_p
        bayes_prior = beta.clone().numpy()
        bayes_prior[salt] = np.abs(np.random.normal(loc=grn_noise_mean, scale=grn_noise_var, size=np.sum(salt)))
        if normalize_mask:
            bayes_prior = bayes_prior/np.max(bayes_prior)
        bayes_prior = torch.tensor(bayes_prior)

elif data_source == "create_data_scMultiSim":
    from scMultiSim import generate_data
    # create synthetic data
    ## parameters for data creation with prefix data
    print("Setting data creation parameters...")
    
    data_cache_path = MODELS_PATH/"scMultiSim_cache"
    data_intervention_type="Cas9"
    data_generate_graph = True
    
    masking_parameters = {}
    create_mask = False
    if masking_mode != None:
        create_mask = True
        with open(parameters_path, "rb") as rf:
            masking_parameters = pickle.load(rf)

    print("Creating synthetic data...")
    mask_precision, mask, samples, gt_interv, sim_regime, beta = generate_data(
            n_genes = data_n_genes,
            grn_params = {"sparsity":0.8},
            n_samples_control = data_n_samples_control,
            n_samples_per_pert = data_n_samples_per_perturbation,
            train_gene_ko = data_train_gene_ko,
            test_gene_ko = data_test_gene_ko,
            pert_type = data_intervention_type,
            pert_strength = 0,
            cache_path = data_cache_path,
            scMultiSim = scMultiSim_path,
            create_mask = create_mask,
            normalize=True,
            pseudocounts = True,
            generate_graph=data_generate_graph,
            graph_kwargs=data_graph_kwargs,
            verbose = False,
            sem = data_sem
    )
    # save data for later validation
    print(f"Saving data at {str(MODELS_PATH.joinpath('synthetic_data'))}...")
    MODELS_PATH.joinpath("synthetic_data").mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_samples.npy"), samples)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_regimes.npy"), sim_regime)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_beta.npy"), beta)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_gt_interv.npy"), gt_interv)

    # initialize data loaders
    print("Initializing Dataloaders...")
    samples = torch.tensor(samples, dtype= float)
    gt_interv = torch.tensor(gt_interv, dtype= int)
    sim_regime = torch.tensor(sim_regime, dtype= int)
    beta = torch.tensor(beta, dtype= float)

    train_loader, validation_loader, test_loader = create_loaders(
        samples=samples, # test_loader is None
        sim_regime=sim_regime,
        validation_size=validation_size,
        batch_size=batch_size,
        SEED= SEED,
        train_gene_ko=data_train_gene_ko,
        test_gene_ko=data_test_gene_ko,
        persistent_workers=False,
        covariates=None,
        num_workers= n_workers,

    )
    model_n_genes = data_n_genes
    model_n_samples = samples.shape[0]
    model_train_gene_ko = data_train_gene_ko
    model_test_gene_ko = data_test_gene_ko
    model_init_tensors = {}
    if masking_loss:
        template = beta.clone().numpy()
        if use_atac_mask:
            template = mask
        values = template[template>0]
        grn_noise_var = np.std(values)
        grn_noise_mean = np.mean(values)
        grn_noise_p = get_sparsity(template)*grn_noise_factor
        salt = np.random.rand(*template.shape) < grn_noise_p
        bayes_prior = template
        bayes_prior[salt] = np.abs(np.random.normal(loc=grn_noise_mean, scale=grn_noise_var, size=np.sum(salt)))
        if normalize_mask:
            bayes_prior = bayes_prior/np.max(bayes_prior)
        bayes_prior = torch.tensor(bayes_prior)
    if use_hard_mask:
        # create hard mask to limit interactions
        hard_mask = bayes_prior > 0

elif data_source == "scMultiSim":
    # get data from data_path
    # to define: samples (samples x genes), gt_interv (genes x contexts), sim_regime (intervened_variables = gt_interv[:, sim_regime].transpose(0, 1)), beta (n_genes x n_genes)
    raise DeprecationWarning("scMultiSim as data source is deprecated! Use 'create_data_scMultiSim instead!")
    # get gt_beta
    sms_path = DATA_PATH/"scMultiSim_data"/data_id
    parameters_path = DATA_PATH/"parameters" / (parameter_set + ".pickle")
    grn = pd.read_csv(sms_path/"unperturbed_data"/"geff.csv", index_col=0).T
    TFs = grn.columns.to_list()
    rna = sc.read_h5ad(sms_path/"ready_full_rna.h5ad")
    atac = sc.read_h5ad(sms_path/"ready_full_atac.h5ad")
    region_to_gene = pd.read_csv(sms_path/"unperturbed_data"/"region_to_gene.csv", index_col=0).to_numpy()
    region_to_tf = pd.read_csv(sms_path/"unperturbed_data"/"region_to_tf.csv", index_col=0).to_numpy()
    with open(parameters_path, "rb") as rf:
        masking_parameters = pickle.load(rf)
    # pad geff matrix with non tf genes and transpose
    beta = np.empty((grn.shape[0], grn.shape[0]))
    for n, (gene_name, row) in enumerate(grn.iterrows()):
        if str(gene_name) in TFs:
            beta[n] = grn[str(gene_name)]
        else:
            beta[n] = np.zeros(grn.shape[0])
    if use_hard_mask:
        # create hard mask to limit interactions
        possible_interactions = np.absolute(region_to_gene.T @ region_to_tf)>0
        possible_interactions=pd.DataFrame(possible_interactions, columns=TFs, index=grn.index)
        # pad with non tf genes and transpose
        hard_mask = np.empty((possible_interactions.shape[0], possible_interactions.shape[0]))
        for n, (gene, row) in enumerate(possible_interactions.iterrows()):
            if str(gene) in TFs:
                hard_mask[n] = possible_interactions[str(gene)]
            else:
                hard_mask[n] = np.zeros(possible_interactions.shape[0])
        # limit self interactions
        hard_mask*=np.diag(np.zeros(hard_mask.shape[0]))

    # get dataloaders
    dataloaders, gt_interv, sim_regime, mask = format_data(
        rna=rna,
        atac=atac,
        gene_names=grn.index.to_list(),
        TFs=TFs,
        region_to_gene=region_to_gene,
        region_to_tf=region_to_tf,
        masking_parameters=masking_parameters,
        batch_size=batch_size,
        validation_size=validation_size,
        device=data_device,
        seed=SEED,
        num_workers=n_workers,
        persistent_workers=False,
        traditional=trad_loading
        )

    if validation_size>0:
        train_loader,validation_loader = dataloaders
    else:
        train_loader = dataloaders[0]
    model_n_genes = rna.shape[1]
    model_n_samples = rna.shape[0]
    model_train_gene_ko = None
    model_test_gene_ko = None
    if masking_mode == "init":
        model_init_tensors = {"beta" : mask}
    else:
        model_init_tensors = {}

    if masking_loss:
        values = grn.to_numpy()[grn>0]
        grn_noise_var = np.std(values)
        grn_noise_mean = np.mean(values)
        grn_noise_p = get_sparsity(beta)*grn_noise_factor
        salt = np.random.rand(*beta.shape) < grn_noise_p
        bayes_prior = beta.copy()
        bayes_prior[salt] = np.abs(np.random.normal(loc=grn_noise_mean, scale=grn_noise_var, size=np.sum(salt)))
        if normalize_mask:
            bayes_prior = bayes_prior/np.max(bayes_prior)
        bayes_prior = torch.tensor(bayes_prior)

elif data_source == "create_data_from_grn":
    # create synthetic data
    ## parameters for data creation with prefix data
    print("Setting data creation parameters...")
    
    grn_path = DATA_PATH/"bicycle_data"/data_id
    for p in grn_path.iterdir():
        if p.name.endswith("GRN.csv"):
            grn_path /= p.name
    beta = pd.read_csv(grn_path, index_col=0).to_numpy()
    data_n_genes=beta.shape[0]
    data_n_samples_control = 500
    data_n_samples_per_perturbation = 250
    data_train_gene_ko=[str(s) for s in range(data_n_genes)]
    data_test_gene_ko = []
    while len(data_test_gene_ko)<= data_n_genes*2:
        data_test_gene_ko.append(tuple(np.random.choice(data_n_genes, size=2, replace=False).astype(str)))
        data_test_gene_ko=list(set(data_test_gene_ko))
    data_test_gene_ko = [f"{x[0]},{x[1]}" for x in data_test_gene_ko]
    data_make_counts=True
    data_verbose=True
    data_intervention_type="Cas9"
    data_T=1.0
    data_library_size_range=[10 * data_n_genes, 100 * data_n_genes]
    

    print("Creating synthetic data...")
    gt_dyn, intervened_variables, samples, gt_interv, sim_regime, beta=create_data_from_grn(
        beta=beta,
        n_samples_control=data_n_samples_control,
        n_samples_per_perturbation=data_n_samples_per_perturbation,
        train_gene_ko=data_train_gene_ko,
        test_gene_ko=data_test_gene_ko,
        verbose=data_verbose,
        device=data_device,
        T=data_T,
        sem=data_sem,
        intervention_type=data_intervention_type,
        make_counts=data_make_counts,
        library_size_range=data_library_size_range,
        graph_kwargs=data_graph_kwargs,
        )

    # save data for later validation
    print(f"Saving data at {str(MODELS_PATH.joinpath('synthetic_data'))}...")
    MODELS_PATH.joinpath("synthetic_data").mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_samples.npy"), samples)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_regimes.npy"), sim_regime)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_beta.npy"), beta)
    np.save(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_gt_interv.npy"), gt_interv)
    

    # initialize data loaders
    print("Initializing Dataloaders...")
    
    train_loader, validation_loader, test_loader = create_loaders(
        samples=samples, # test_loader is None
        sim_regime=sim_regime,
        validation_size=validation_size,
        batch_size=batch_size,
        SEED= SEED,
        train_gene_ko=data_train_gene_ko,
        test_gene_ko=data_test_gene_ko,
        persistent_workers=False,
        covariates=None,
        num_workers= n_workers,
    )
    model_n_genes = samples.shape[1]
    model_n_samples = samples.shape[0]
    model_train_gene_ko = data_train_gene_ko
    model_test_gene_ko = data_test_gene_ko
    model_init_tensors = {}
    if masking_loss:
        values = beta.numpy()[beta>0]
        grn_noise_var = np.std(values)
        grn_noise_mean = np.mean(values)
        grn_noise_p = get_sparsity(beta)*grn_noise_factor
        salt = np.random.rand(*beta.shape) < grn_noise_p
        bayes_prior = beta.clone()
        bayes_prior[salt] = np.abs(np.random.normal(loc=grn_noise_mean, scale=grn_noise_var, size=np.sum(salt)))
        if normalize_mask:
            bayes_prior = bayes_prior/np.max(bayes_prior)
        bayes_prior = torch.tensor(bayes_prior)

else:
    raise ValueError(f"Invalid data_source: {data_source}. Choose one of ['create_data','create_data_from_grn','scMultiSim']!")

# initialize parameters for the bicycle class with prefix model
print("Initializing model parameters...")

model_device = torch.device(f"cuda:{GPU_DEVICES[0]}")
model_gt_interv = gt_interv.to(model_device)
model_lyapunov_penalty = True
model_perfect_interventions = True
model_rank_w_cov_factor = model_n_genes
model_optimizer = "adam"
# Faster decay for estimates of gradient and gradient squared:
model_optimizer_kwargs = {"betas": [0.5, 0.9]}
model_early_stopping = False
model_early_stopping_min_delta = 0.01
model_early_stopping_patience = 500
model_early_stopping_p_mode = True
model_x_distribution = "Multinomial"
model_x_distribution_kwargs = {}

model_mask = get_diagonal_mask(n_genes=model_n_genes, device=model_device)
if use_hard_mask:
    model_mask=torch.tensor(hard_mask, device=model_device)
elif masking_loss:
    model_mask = None
model_use_encoder = False
model_gt_beta = beta.to(model_device)

model_covariates = None
model_n_factors = 0
model_intervention_type = "Cas9"
model_learn_T = False
model_train_only_likelihood = False
model_train_only_latents = False
model_mask_genes = []

# training variables
swa = 100                                         # hyperparameter
# how often to caculate nll for validation
check_val_every_n_epoch = 100
log_every_n_steps = 100

def train_bicycle(config):
    
    empty_report = {
        "final_nll":np.nan,
        "final_max_f1":np.nan,
        "final_average_precision":np.nan,
        "final_auroc":np.nan,
        "nll":np.nan,
        "average_precision":np.nan,
        "avg_train_loss":np.nan,
        "avg_valid_loss":np.nan,
        }

    context = tune.get_context()
    trial_dir = tuner_path.glob("train_bicycle*").__next__()/Path(context.get_trial_dir()).name
    print(f"\n\n Trial directory: {trial_dir}")
    tune.report(empty_report)
    
    samples = np.load(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_samples.npy"))
    sim_regime = np.load(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_regimes.npy"))
    beta = np.load(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_beta.npy"))
    gt_interv = np.load(os.path.join(MODELS_PATH.joinpath("synthetic_data"), "check_sim_gt_interv.npy"))

    # initialize data loaders
    print("Initializing Dataloaders...")
    samples = torch.tensor(samples, dtype= float)
    gt_interv = torch.tensor(gt_interv, dtype= int)
    sim_regime = torch.tensor(sim_regime, dtype= int)
    beta = torch.tensor(beta, dtype= float)

    train_loader, validation_loader, test_loader = create_loaders(
        samples=samples, # test_loader is None
        sim_regime=sim_regime,
        validation_size=validation_size,
        batch_size=batch_size,
        SEED= SEED,
        train_gene_ko=data_train_gene_ko,
        test_gene_ko=data_test_gene_ko,
        persistent_workers=False,
        covariates=None,
        num_workers= n_workers,
    )


    # assign id to run
    
    gradient_clip_val = config.get("gradient_clip_val", 1.0)
    sigma_min = config.get("sigma_min", 1e-3)
    scale_l1 = config.get("scale_l1", 0.1)
    scale_spectral = config.get("scale_spectral", 1.0)
    scale_lyapunov = config.get("scale_lyapunov", 1.0)
    scale_kl = config.get("scale_kl", 1.0)
    scale_mask = config.get("scale_mask", 1.0)
    model_T = config.get("model_T", 1.0)
    lr = config.get("lr", 1e-3)
    print("Initializing BICYCLE model...")
    model = BICYCLE(
        lr = lr,
        gt_interv = model_gt_interv,
        n_genes = model_n_genes,
        n_samples = model_n_samples,
        lyapunov_penalty = model_lyapunov_penalty,
        perfect_interventions = model_perfect_interventions,
        rank_w_cov_factor = model_rank_w_cov_factor,
        optimizer = model_optimizer,
        optimizer_kwargs = model_optimizer_kwargs,
        device = model_device,
        scale_l1 = scale_l1,
        scale_spectral = scale_spectral,
        scale_lyapunov = scale_lyapunov,
        scale_kl = scale_kl,
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
        sigma_min = sigma_min,
        T = model_T,
        learn_T = model_learn_T,
        train_only_likelihood = model_train_only_likelihood,
        train_only_latents = model_train_only_latents,
        mask_genes = model_mask_genes,
        bayes_prior=None if not masking_loss else bayes_prior.to(model_device),
        scale_mask=0 if not masking_loss else scale_mask,
        hamming_distance=bin_prior,
        metrics_to_report=empty_report
    )
    
    torch.compile(model=model)
    #os.environ["TORCH_USE_CUDA_DSA"] = True
    model.to(model_device)

    del samples, sim_regime, beta, gt_interv
    print("Training data:")
    print(f"- Number of training samples: {len(train_loader.dataset)}")
    
    # callback variables
    loggers = DictLogger()

    # initialize training callbacks
    callbacks = []
    if swa > 0:
        callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))

    if early_stopping:
        callbacks.append(EarlyStopping(monitor="valid_loss", mode="min", patience=300))

    # pretraining
    #callbacks.append(RayTrainReportCallback())
    if model.use_latents and n_epochs_pretrain_latents > 0:
        print(f"Pretraining model latents for {n_epochs_pretrain_latents} epochs...")
        pretrain_callbacks = []

        if swa > 0:
            pretrain_callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))

        #pretrain_callbacks.append(RayTrainReportCallback())
        pretrainer = pl.Trainer(
            devices=model_device,
            accelerator="gpu",
            #strategy=RayDDPStrategy(),
            #plugins=[RayLightningEnvironment()],
            max_epochs=n_epochs_pretrain_latents,
            logger=loggers,
            log_every_n_steps=log_every_n_steps,
            enable_model_summary=False,
            enable_progress_bar=False,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=0,
            callbacks=pretrain_callbacks,
            gradient_clip_val=gradient_clip_val,
            default_root_dir=trial_dir,
            gradient_clip_algorithm="norm",
            deterministic= "warn",
        )
        #pretrainer = prepare_trainer(pretrainer)

        print("PRETRAINING LATENTS!")
        start_time = time.time()
        model.train_only_likelihood = True # TODO: create alternative option for pretraining
        pretrainer.fit(model, train_loader, validation_loader)
        end_time = time.time()
        model.train_only_likelihood = False
        pretraining_time = float(end_time - start_time)
        print(f"Pretraining took {pretraining_time} seconds.")
        del pretrainer
    print(f"Training the model for {n_epochs} epochs...")
    # training
    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        #strategy=RayDDPStrategy(),
        #plugins=[RayLightningEnvironment()],
        max_epochs=n_epochs,
        logger=loggers,
        log_every_n_steps=log_every_n_steps,
        enable_model_summary=False,
        enable_progress_bar=False,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        default_root_dir=trial_dir,
        gradient_clip_algorithm="norm",
        deterministic= "warn",
    )
    #trainer = prepare_trainer(trainer)


    print("TRAINING THE MODEL!")

    start_time = time.time()
    trainer.fit(model, train_loader, validation_loader)
    end_time = time.time()
    training_time = float(end_time - start_time)
    print(f"Training took {training_time} seconds.")

    nll, max_f1, average_precision, auroc, prior_average_precision, prior_auroc = model.evaluate(test_loader.dataset)
    empty_report.update({
        "final_nll":nll.cpu().numpy(),
        "final_max_f1":max_f1.cpu().numpy(),
        "final_average_precision":average_precision.cpu().numpy(),
        "final_auroc":auroc.cpu().numpy(),
        }
        )
    if masking_loss:
        empty_report.update({
            "prior_average_precision": prior_average_precision.cpu().numpy(), 
            "prior_auroc": prior_auroc.cpu().numpy(),
        })

    tune.report(empty_report)
    del nll, max_f1, average_precision, auroc, prior_average_precision, prior_auroc
    del trainer, model, train_loader, validation_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

ray.init()
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
#os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_memory_usage_threshold"] = str(0.99)


metric = "average_precision"
mode = "min"
if metric != "nll":
    mode = "max"
time_budget = timedelta(hours=10)
tuner_path = MODELS_PATH/"tune_results"
tuner_path.mkdir()

search_space={
    #"gradient_clip_val" : tune.uniform(0, 1.0),
    #"sigma_min" : tune.loguniform(0.001, 0.1),
    "lr" : tune.loguniform(0.01, 10),
    "scale_l1" : tune.loguniform(0.01, 10),
    "scale_spectral" : tune.loguniform(0.01, 10),
    "scale_lyapunov" : tune.loguniform(0.01, 10),
    "scale_kl" : tune.loguniform(0.01, 10),
    #"model_T" : 1.0,
    }

if masking_loss:
    search_space["scale_mask"] = tune.loguniform(0.01, 10)

run_config = RunConfig(
    storage_path=tuner_path,
    log_to_file=(tuner_path/"stdout.log", tuner_path/"stderr.log"),
)

bayes_opt_search = BayesOptSearch(metric=metric, mode = mode)

def trial_name_creator(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"

tune_config=tune.TuneConfig(
    num_samples=100,
    scheduler=ASHAScheduler(grace_period=10, stop_last_trials=False),
    search_alg=bayes_opt_search,
    max_concurrent_trials=4,
    metric=metric,
    mode = mode, 
    reuse_actors=False,
    trial_dirname_creator= trial_name_creator,
    #time_budget_s=time_budget
)

train_with_recources = tune.with_resources(train_bicycle, resources = {"gpu":len(GPU_DEVICES)/tune_config.max_concurrent_trials})

tuner = tune.Tuner(
    trainable=train_with_recources,
    param_space={"train_loop_config": search_space},
    tune_config=tune_config,
    run_config=run_config,
)
results = tuner.fit()


results.get_dataframe(filter_metric=metric).to_csv(MODELS_PATH/"results_df.csv")
dfs = {result.path: result.metrics_dataframe for result in results}
try:
    for result in results:
        print(result.path/"result.csv")
        result.metrics_dataframe.to_csv(result.path/"result.csv")
except:
    print("ERROR when saving result DataFrames")
# log the environment
try:
    best_result = results.get_best_result(metric="final_" + metric, mode=mode)
    print(best_result)
    best_result.metrics_dataframe.to_csv(MODELS_PATH/"best_metrics.csv")
except RuntimeError:
    print("No best trial found!")
pd.DataFrame(globals().items()).to_csv(MODELS_PATH/"globals.csv")

ray.shutdown()