
import anndata
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from bicycle.utils.mask_utils import get_mask, string_to_list

def format_data(
        rna:anndata.AnnData,
        atac:anndata.AnnData,
        gene_names: list,
        TFs:list,
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
    atac_unp=np.array(atac.X[~atac.obs.perturbed])
    naked_mask = get_mask(
        atac = atac_unp.T,
        region_to_gene=region_to_gene,
        region_to_tf=region_to_tf,
        **masking_parameters,)
    #pad and transpose the mask
    mask = np.empty((naked_mask.shape[0], naked_mask.shape[0]))
    for n, gene_name in enumerate(gene_names):
        if str(gene_name) in TFs:
            mask[n] = naked_mask[:, TFs.index(str(gene_name))]
        else:
            mask[n] = np.zeros(naked_mask.shape[0])
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
        ) if n ==0 else DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        ) for n,  dataset in enumerate(datasets) ]

    return dataloaders, gt_interv, sim_regime, mask
