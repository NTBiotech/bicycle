import pandas as pd
from pathlib import Path
import numpy as np
import scipy.stats as stats
import anndata
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def above_threshold(matrix:np.array, percentile : int = 50, threshold : int = None):
    if threshold is None:
        threshold = np.percentile(matrix, percentile)
    return (matrix >= np.full(shape=matrix.shape,fill_value=threshold)).astype(int)

def normalize_matrix(df: np.array):
    if type(df) == np.ndarray:
        print("np.array")
        return df/sum(df)
    else:
        return df/df.sum().sum()

def distance(x1:np.array, x2: np.array):
    x1 = np.asarray(x1, dtype=np.float32)
    x2 = np.asarray(x2, dtype=np.float32)
    return np.sqrt(np.sum(np.square(np.subtract(x1, x2))))

def get_sparsity(mtx:np.ndarray):
    return np.count_nonzero(mtx)/np.prod(mtx.shape)

def add_noise(mtx:np.ndarray, mean=1, std=0.1):
    return mtx * np.random.normal(loc=mean, scale=std, size=mtx.shape)

def add_saltpepper(mtx:np.ndarray, p, min=None, max=None):
    if max is None:
        max = np.max(mtx)    
    if min is None:
        min = np.min(mtx)
    indexes = np.random.rand(*mtx.shape)<p
    mtx[indexes] = np.random.choice([min, max], p=[0.5, 0.5], size=np.sum(indexes))
    return mtx

def get_random_samples(n_samples, model: np.ndarray):
    sparsity = get_sparsity(model)
    size = model.shape
    # make binary atac
    binary = np.random.rand(n_samples, *size) < sparsity
    # fill binary with values
    uniques, counts = np.unique(model[model>0], return_counts=True)
    p = counts/counts.sum()

    values = np.random.choice(uniques, p=p, size=binary.sum())
    random_samples = np.zeros(shape=(n_samples, *size))
    random_samples[binary] = values
    return random_samples

def evaluate_model(grn: np.array, atac: np.array, get_mask, funct_params: dict):
    """Evaluate the get_mask function based on the distance of its output to the grn."""
    mask = get_mask(atac, **funct_params)
    mask_distance = distance(grn, mask)
    # distance distribution when atac is random
    max = np.max(atac)
    size = atac.shape
    n_samples = 10000
    rand_distances = []
    rand_atac = np.random.normal(0, max, size=(n_samples,)+size)

    for atac in rand_atac:
        mask = get_mask(mask, **funct_params)
        rand_distances.append(distance(grn, mask))

    p_value = stats.percentileofscore(rand_distances, mask_distance)

    print("Distance to grn: ", mask_distance)
    print("p-value:", stats.percentileofscore(rand_distances, mask_distance))
    return mask_distance, p_value

def get_mask(atac,
             region_to_gene,
             region_to_tf,
             threshold = False,
             threshold_kwargs = {},
             correlation = False,
             correlation_kwargs = {},
             pseudocounts = False,
             comment=None,
             ):
    """
    Args:
    atac (np.array): regions x samples
    params (dict): function parameters
    """
    if pseudocounts:
        atac += np.min(atac)*0.0001
    if correlation:
        corr_atac = np.abs(np.corrcoef(atac))
        if type(correlation_kwargs.get("mask", False)) is np.ndarray:
            corr_atac *= correlation_kwargs["mask"]
        if correlation_kwargs.get("normalize", False):
            corr_atac = normalize_matrix(corr_atac)
        if correlation_kwargs.get("threshold", False):
            corr_atac = above_threshold(corr_atac, **correlation_kwargs["threshold_kwargs"])
    else:
        corr_atac = atac @ atac.T
    mask = region_to_gene.T @ corr_atac @ region_to_tf
    if threshold:
        mask = above_threshold(mask, **threshold_kwargs)
    return mask
def get_mask_no_data(region_to_gene, region_to_tf, threshold = False, threshold_kwargs = {}, correlation = False, correlation_kwargs = {}, pseudocounts = False, comment=None):
    
    mask = region_to_gene.T @ region_to_tf
    if threshold:
        mask = above_threshold(mask, **threshold_kwargs)
    return mask.astype(np.float32)

def string_to_list(string:str, to_type=int):
    """
    Function to convert lists in string-format into python lists.
    Relevant for string-format lists in Anndata objects.
    """
    string = string[1:-1]
    list = string.split(",")
    list = [to_type(x) for x in list]
    return list

def format_data(
        rna:anndata.AnnData,
        atac:anndata.AnnData,
        region_to_gene,
        region_to_tf,
        masking_parameters:dict,
        batch_size,
        validation_size:float =0.2,
        device=torch.device("cpu"),
        seed = 0,
        num_workers = 1,
        persistent_workers=False,
        traditional:bool=False,
        ):
    """Function to create data loaders for bicycle.
    Args:
    - rna (Anndata): scRNA data with self.obs.perturbed (bool) and self.obs.target_genes (Series of string-format lists with "[""]" for no perturbation)
        to specify perturbation context
    - atac (Anndata): scATAC data with self.obs.perturbed (bool) and self.obs.target_genes (Series of string-format lists with "[""]" for no perturbation)
        to specify perturbation context
    Note:
    - The training dataloader contains the validation data, to learn the latents.
    """
    
    # creating a mask
    atac_unp=atac.X[~atac.obs.perturbed]
    mask = get_mask(
        atac = atac_unp.T,
        region_to_gene=region_to_gene,
        region_to_tf=region_to_tf,
        **masking_parameters,)
    mask = torch.Tensor(mask, device=device)

    contexts = (rna.obs.target_genes).unique()
    target_to_gtinterv_col = {contexts[n]:n for n in range(len(contexts))}

    # convert contexts to list objects
    contexts = [string_to_list(x, to_type=str) for x in contexts]
    # number contexts (0:len(contexts)] and create gt_interv with genes x context
    genes = rna.var_names.to_numpy()
    gt_interv = torch.Tensor(np.array([[g in c for g in genes] for c in contexts]).T)
    # data for dataloaders
    sim_regime = torch.Tensor([target_to_gtinterv_col[target] for target in rna.obs.target_genes]).long()
    samples = torch.Tensor(np.array(rna.X),)
    index = torch.arange(0, len(rna.obs_names))

    if not traditional:
        # create datasets
        datasets = [TensorDataset(
            samples,
            torch.Tensor(sim_regime,),
            index
        )]

        if validation_size >0:
            train_index, val_index = train_test_split(
                index,
                test_size=validation_size,
                random_state=seed,
                shuffle=True,
                stratify=sim_regime
            )
            datasets.append(TensorDataset(
                samples[val_index],
                sim_regime[val_index],
                val_index
            ))
    else:
        
        if validation_size > 0:

            train_index, val_index = train_test_split(
                index,
                test_size=validation_size,
                random_state=seed,
                shuffle=True,
                stratify=sim_regime
            )
            datasets = [TensorDataset(
                torch.cat((samples[train_index],
                          samples[val_index]), dim=0),
                torch.cat((sim_regime[train_index],
                          sim_regime[val_index]), dim=0),
                torch.cat((train_index,
                          val_index), dim=0),
                torch.cat((torch.zeros(train_index.shape[0]),
                          torch.ones(val_index.shape[0])), dim=0),
            )]

            datasets.append(TensorDataset(
                samples[val_index],
                sim_regime[val_index],
                val_index,
                torch.ones(val_index.shape[0])
            ))

        else:
            datasets = [TensorDataset(
            samples,
            torch.Tensor(sim_regime),
            index,
            np.zeros(index.shape[0])
        )]
    # create dataloaders
    dataloaders = [DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        ) for dataset in datasets]

    return dataloaders, gt_interv, sim_regime, mask
