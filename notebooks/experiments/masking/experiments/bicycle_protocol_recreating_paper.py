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

replicates = 5

DATA_PATH = Path("/data/toulouse/bicycle/notebooks/experiments/masking/data/model_runs/experiments/recreating_paper")
# add a copy of the protocoll to the dest path
#shutil.copy(__file__, DATA_PATH)

SEED = np.random.randint(0,100)
GPU_DEVICES = [0]

iterator1 = ["linear-ou", "linear"]
iterator2 = np.arange(2,12, 2)
for n in iterator1:
    for n2 in iterator2:
        for r in range(replicates):
            DATA_PATH = Path("/data/toulouse/bicycle/notebooks/experiments/masking/data/model_runs/experiments/recreating_paper")


            monitor_stats = False
            profile = False
            CHECKPOINTING = True
            early_stopping = False
            progressbar_rate = 0
            # model
            compile = True
            compiler_kwargs = {}
            compiler_fullgraph = False
            compiler_dynamic = False
            compiler_mode = None
            loader_workers=1
            trainer_precision=32
            matmul_precision="high"
            trad_loading=True

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
            data_sem=n

            # scMultiSim
            scMultiSim_path = DATA_PATH.parent/"scMS.R"
            data_id = "data_run000"

            # needs to be a round number
            data_n_genes = 10
            # need to be greater than 1
            data_n_samples_control = 500
            data_n_samples_per_perturbation = 250
            data_verbose=False

            # training
            model_use_latents = False    # determines if latent representation is learned

            validation_size = 0.2

            batch_size = 10000
            n_epochs = 5000
            n_epochs_pretrain_latents = 1000

            # testing
            evaluate = False

            if masking_mode != "loss" and bin_prior:
                raise NotImplementedError("masking mode must be loss for bin_prior")
            if trad_loading:
                from bicycle.model import BICYCLE
            else:
                from model import BICYCLE
            masking_loss = masking_mode == "loss"
            use_hard_mask = masking_mode == "hard"

            torch.set_float32_matmul_precision(matmul_precision)


            # configure output paths
            ## subdirectories model and plots
            ## in subdirectory data of the current path
            parameters_path = DATA_PATH/"parameters" / (parameter_set + ".pickle")


            print("Creating output directories...")

            MODELS_PATH = DATA_PATH
            # assign id to run
            run_id = get_id(MODELS_PATH, prefix="run_", id_length=3)

            MODELS_PATH = MODELS_PATH/run_id
            if not MODELS_PATH.exists():
                MODELS_PATH.mkdir(parents=True, exist_ok=True)

            print(f"Output path is: {str(MODELS_PATH)}!")


            if not trad_loading:
                validation_size=0
            data_device=torch.device("cpu")


            data_train_gene_ko=[str(s) for s in range(data_n_genes)][:n2]
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
                    verbose = data_verbose,
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
                    num_workers= loader_workers,

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
                        mask_kwargs = masking_parameters,
                        generate_graph=data_generate_graph,
                        graph_kwargs=data_graph_kwargs,
                        verbose = data_verbose,
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
                print(samples.shape)
                gt_interv = torch.tensor(gt_interv, dtype= int)
                print(gt_interv.shape)
                sim_regime = torch.tensor(sim_regime, dtype= int)
                print(sim_regime.shape)
                beta = torch.tensor(beta, dtype= float)
                print(beta.shape)

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
                    num_workers= loader_workers,

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

            elif data_source == "scMultiSim":
                if evaluate:
                    raise NotImplementedError("Testing on scMultiSim data not implemented!")
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
                    num_workers=loader_workers,
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
                    num_workers= loader_workers,
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
            model_lr = 1e-3
            model_gt_interv = gt_interv
            model_lyapunov_penalty = True
            model_perfect_interventions = True
            model_rank_w_cov_factor = model_n_genes
            model_optimizer = "adam"
            # Faster decay for estimates of gradient and gradient squared:
            model_optimizer_kwargs = {"betas": [0.5, 0.9]}
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
            model_device = torch.device(f"cuda:{GPU_DEVICES[0]}")

            model_mask = get_diagonal_mask(n_genes=model_n_genes, device=model_device)
            if use_hard_mask:
                model_mask=torch.tensor(hard_mask, device=model_device)
            elif masking_loss:
                model_mask = None
            model_use_encoder = False
            model_gt_beta = beta

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
            if trad_loading:
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
                    bayes_prior=None if not masking_loss else bayes_prior.to(model_device),
                    scale_mask=0 if not masking_loss else scale_mask,
                    hamming_distance=bin_prior
                )
            else:
                if masking_loss:
                    raise NotImplementedError("Masking loss and not trad_loading not implemented yet!")
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
                    loss_scale = torch.tensor([model_scale_l1, model_scale_spectral, model_scale_lyapunov, model_scale_kl]),
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
                    norm_scale = True
                )
            if compile:
                torch.compile(model=model, 
                    fullgraph = compiler_fullgraph,
                    dynamic = compiler_dynamic,
                    mode=compiler_mode)
            model.to(model_device)


            # training variables
            gradient_clip_val = 1.0
            plot_epoch_callback = 500 # intervall for plot_training_results -> GenerateCallback

            print("Training data:")
            print(f"- Number of training samples: {len(train_loader.dataset)}")
            if validation_size > 0:
                print(f"- Number of validation samples: {len(validation_loader.dataset)}")

            # callback variables
            swa = 0
            VERBOSE_CHECKPOINTING = False
            check_val_every_n_epoch = 1
            log_every_n_steps = 1

            if profile:
                # initialize profiler to find bottlenecks
                MODELS_PATH.joinpath("profiler").mkdir(parents=True, exist_ok=True)
                profiler = AdvancedProfiler(dirpath=MODELS_PATH.joinpath("profiler"), filename="profile")

            # initialize logger for the Trainer class
            #if not (MODELS_PATH/"logs").exists():
            #    (MODELS_PATH/"logs").mkdir(parents=True)
            loggers = DictLogger()

            # initialize training callbacks
            MODELS_PATH.joinpath("generatecallback").mkdir(parents=True, exist_ok=True)
            callbacks = []
            if monitor_stats:
                callbacks.append(DeviceStatsMonitor())
            if swa > 0:
                callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))

            if CHECKPOINTING:
                MODELS_PATH.joinpath("customcheckpoint").mkdir(parents=True, exist_ok=True)
                callbacks.append(
                    CustomModelCheckpoint(
                        dirpath=os.path.join(MODELS_PATH, "customcheckpoint"),
                        filename="{epoch}",
                        save_last=True,
                        save_top_k=1,
                        verbose=VERBOSE_CHECKPOINTING,
                        monitor="valid_loss",
                        mode="min",
                        save_weights_only=True,
                        start_after=100,
                        save_on_train_epoch_end=False,
                        every_n_epochs=1,
                    )
                )
                MODELS_PATH.joinpath("mylogger").mkdir(parents=True, exist_ok=True)
                callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODELS_PATH, "mylogger")))
            if early_stopping:
                if not trad_loading:
                    raise NotImplementedError("early stopping only available for trad_loading")
                callbacks.append(EarlyStopping(monitor="valid_loss", mode="min", patience=300))

            # pretraining

            if model.use_latents and n_epochs_pretrain_latents > 0:
                print(f"Pretraining model latents for {n_epochs_pretrain_latents} epochs...")
                pretrain_callbacks = []
                if monitor_stats:
                    pretrain_callbacks.append(DeviceStatsMonitor())

                if swa > 0:
                    pretrain_callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))

                if CHECKPOINTING:
                    MODELS_PATH.joinpath("mylogger").mkdir(parents=True, exist_ok=True)
                    pretrain_callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODELS_PATH, "mylogger")))

                if profile:
                    profiler.filename = "pretraining_profile"

                pretrainer = pl.Trainer(
                    max_epochs=n_epochs_pretrain_latents,
                    accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
                    logger=loggers,
                    log_every_n_steps=log_every_n_steps,
                    enable_model_summary=True,
                    enable_progress_bar=True,
                    enable_checkpointing=CHECKPOINTING,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    devices=GPU_DEVICES,  # if str(device).startswith("cuda") else 1,
                    num_sanity_val_steps=0,
                    callbacks=pretrain_callbacks,
                    gradient_clip_val=gradient_clip_val,
                    default_root_dir=str(MODELS_PATH),
                    gradient_clip_algorithm="value",
                    deterministic=False,  # "warn",
                    profiler=profiler if profile else None,
                    precision=trainer_precision,
                )
                print("PRETRAINING LATENTS!")
                start_time = time.time()
                model.train_only_likelihood = True # TODO: create alternative option for pretraining
                if validation_size>0:
                    pretrainer.fit(model, train_loader, validation_loader)
                else:
                        pretrainer.fit(model, train_loader)
                end_time = time.time()
                model.train_only_likelihood = False
                if profile:
                    profiler.filename = "training_profile"
                pretraining_time = float(end_time - start_time)
                print(f"Pretraining took {pretraining_time} seconds.")

            print(f"Training the model for {n_epochs} epochs...")
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
                devices=GPU_DEVICES,  # if str(device).startswith("cuda") else 1,
                num_sanity_val_steps=0,
                callbacks=callbacks,
                gradient_clip_val=gradient_clip_val,
                default_root_dir=str(MODELS_PATH),
                gradient_clip_algorithm="value",
                deterministic=False,  # "warn",
                profiler=profiler if profile else None,
                precision=trainer_precision,
            )

            print("TRAINING THE MODEL!")

            start_time = time.time()
            if validation_size>0:
                trainer.fit(model, train_loader, validation_loader)
            else:
                trainer.fit(model, train_loader)
            end_time = time.time()

            training_time = float(end_time - start_time)
            print(f"Training took {training_time} seconds.")

            pd.DataFrame(globals().items()).to_csv(MODELS_PATH/"globals.csv")

            if evaluate:
                model.to(data_device)
                if masking_loss:
                    nll, max_f1, average_precision, auroc, prior_average_precision, prior_auc =  model.evaluate(test_loader.dataset)
                    print(f"nll:{nll}, max_f1:{max_f1}, average_precision:{average_precision}, auroc:{auroc}, prior_average_precision:{prior_average_precision}, prior_auc:{prior_auc}")
                else:
                    nll, max_f1, average_precision, auroc =  model.evaluate(test_loader.dataset)
                    print(f"nll: {nll}, max_f1: {max_f1}, average_precision: {average_precision}, auroc: {auroc}")



            # log the environment
            pd.DataFrame(globals().items()).to_csv(MODELS_PATH/"globals.csv")
