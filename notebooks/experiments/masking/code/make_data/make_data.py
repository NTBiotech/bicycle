"""Script for creating scRNA and scATAC perturbation data with scMultiSim"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from scipy.special import expit
from bicycle.utils.data import generate_weighted_graph, generate_grn
from bicycle.utils.general import get_id
from bicycle.utils.plotting import plot_comparison
from bicycle.utils.mask_utils import get_mask2, normalize_data, distance
from sklearn.metrics import average_precision_score
import scanpy as sc
#from torch.distributions import PositiveDefiniteTransform, LowerCholeskyTransform
import torch
import json
from scipy.stats import ttest_rel

os.chdir(Path(__file__).parent)




def generate_data(
        n_genes:int,
        grn_params:dict,
        grn_shape,
        n_samples_control:int = 0,
        n_samples_per_pert:int = 250,
        train_gene_ko:list = [],
        test_gene_ko:list = [],
        pert_type:str = "dCas9",
        pert_strength:float = 0.01,
        add_noise:bool = False,
        add_batch:bool = False,
        cache_path:Path = Path("./scMultiSim_cache/"),
        scMultiSim:Path = Path("./scMultiSim_arg.R"),
        create_mask:bool = False,
        normalize:bool = False,
        pseudocounts = False,
        generate_graph = True,
        mask_kwargs:dict = {},
        graph_kwargs = {},
        verbose = False,
        sem_regime = "pert_grn",

):
    """Function to generate perturbed scRNA data using scMultiSim."""

    def format_for_scMultiSim(grn:np.array, cache_path:Path, pseudocounts:bool = True):
        """Function that formats the grn into a hot encoded form and saves it in the cache path as 'food_for_R.csv'.
        Note:
            - The grn does not have to be symmetric
        """
        if pseudocounts:
            new_grn= grn.copy() + 0.00001
        else:
            new_grn = grn.copy()
        TF_names = np.arange(1, new_grn.shape[1]+1).astype(str)
        gene_names = np.arange(1, new_grn.shape[0]+1).astype(str)
        new_grn = pd.DataFrame(new_grn, columns=TF_names, index=gene_names)
        params= np.empty((np.prod(new_grn.shape), 3))
        n=0
        for gene,row in new_grn.iterrows():
            for tf, effect in row.items():
                params[n] = [gene, tf, effect]
                n+=1

        np.random.shuffle(params)
        params = pd.DataFrame(params, columns=["regulated.gene","regulator.gene","regulator.effect"])
        params=params.loc[params["regulator.effect"]!=0]
        params = params[~params.drop(columns="regulator.effect").duplicated()]

        # formatting for R to understand it
        params["regulated.gene"] = params["regulated.gene"].astype(int)
        params["regulator.gene"] = params["regulator.gene"].astype(int)
        params = params.sort_values("regulator.gene")
        params.index = np.arange(1, len(params)+1).astype(str)
        params.to_csv(cache_path/f"food_for_R.csv", quoting=csv.QUOTE_NONNUMERIC)


    # No intervention must be in train and test sim regimes:
    if len([x for x in train_gene_ko if x in test_gene_ko]) > 0:
        raise ValueError("train and test gene knock-outs must be disjoint")
    

    os.chdir(scMultiSim.parent.as_posix())
    cache_path = Path(cache_path.parent/get_id(cache_path.parent,2,cache_path.name+"_"))
    cache_path.mkdir()
    
    # generate a grn
    print("Generating the GRN...")
    if generate_graph:
        unsuccessful_sampling = True
        while unsuccessful_sampling:
            beta = generate_weighted_graph(
                graph_type="erdos-renyi",
                nodes=n_genes,
                edge_assignment="random-uniform",
                make_contractive=True,
                **graph_kwargs,
            )
            beta = torch.tensor(np.abs(beta)).float()
            # Compute eigenvalues of beta
            B = torch.eye(n_genes) - beta
            eigvals_B = np.real(np.linalg.eigvals(B))
            # Check if all eigvals are positive
            if np.all(eigvals_B > 0):
                unsuccessful_sampling = False
                beta = beta.numpy()
                print(beta)
            else:
                print("*" * 100)
                print("Unsuccessful sampling. Re-sampling...")
                print("*" * 100)
    else:
        beta = generate_grn(
            n_genes=n_genes,
            grn_shape=grn_shape,
            grn_params=grn_params,
            eigenvalues=True,
            distribute_TFs=False,
            out_path=cache_path,
            verbose=verbose,
            **graph_kwargs
            )

    contexts = train_gene_ko + test_gene_ko

    contexts_ctrl = [""] + contexts
    # sim_regime for ctrls
    sim_regime = np.zeros(n_samples_control, dtype=int)
    # sim_regime for all
    sim_regime = np.append(sim_regime, np.repeat(np.arange(1,len(contexts_ctrl), dtype=int), n_samples_per_pert))
    # Genes x regimes matrices, which indexes which genes are intervened in
    # which context
    gt_interv = np.zeros((n_genes, len(contexts_ctrl)))

    # Gene i is intervened in regime i
    for idx, p in enumerate(contexts_ctrl):
        # Empy string represents control
        if p == "":
            continue
        elif "," in p:
            # Comma separated values represent multiple interventions
            for pp in p.split(","):
                gt_interv[int(pp), idx] = 1
        else:
            gt_interv[int(p), idx] = 1


    if sem_regime == "pert_cif":
        # make matrices to multiply with cif matrices
        n_samples = n_samples_control + len(contexts)*n_samples_per_pert
        n_cifs = 50
        shape=(n_samples, n_cifs + n_genes)
        if pert_strength != 0:
            pert_kwargs = {"kon":pert_strength, "koff":1/pert_strength}
        else:
            pert_kwargs = {"kon":pert_strength, "koff":100000}
        for name, factor in pert_kwargs.items():
            cif = np.ones(shape)
            for n, context in enumerate(gt_interv.T[1:]):
                i = n_samples_control + n*n_samples_per_pert
                cif[i : i+n_samples_per_pert, -n_genes:][:, context.astype(bool)] = factor
            np.savetxt(cache_path/f"cif_{name}_mod.csv", cif, delimiter=",")
        intervention = "TRUE"

    elif sem_regime == "pert_grn":
        n_samples = n_samples_control
        intervention = "FALSE"

    else:
        raise NotImplementedError(f"sem_regime '{sem_regime}' not implemented! Please use one of ['pert_grn', 'pert_cif'].")


    if n_samples >0:
        # generate unperturbed data
        format_for_scMultiSim(beta, cache_path, pseudocounts)
        print(os.system(
            f"Rscript {scMultiSim.name} --n_genes {n_genes} --cache {cache_path.as_posix()} --n_samples {n_samples} --noise {add_noise} --batch {add_batch} --intervention {str(intervention).upper()}")
            )

        df = pd.read_csv(cache_path/"R_out"/"counts.csv", index_col=0).T#.iloc[:, :-n_unregulated:]
        #if normalize:
        #    df = normalize_data(df)
        df.to_csv(cache_path/"rna_matrix.csv")

        # generate geff and region matrices
        geff =  pd.read_csv(cache_path/"R_out"/"geff.csv", index_col=0).to_numpy()
        region_to_gene = pd.read_csv(cache_path/"R_out"/"region_to_gene.csv", index_col=0).to_numpy()
        region_to_tf = pd.read_csv(cache_path/"R_out"/"region_to_tf.csv", index_col=0).to_numpy()

        if create_mask:
            # get atac data for masking
            df = pd.read_csv(cache_path/"R_out"/"atac_counts.csv", index_col=0).T
            #if normalize:
            #    df = normalize_data(df)
            df.to_csv(cache_path/"atac_matrix.csv")
    
    else:
        # generate geff and region matrices
        format_for_scMultiSim(beta, cache_path, pseudocounts)
        print(os.system(
            f"Rscript {scMultiSim.name} --n_genes {n_genes} --cache {cache_path.as_posix()} --n_samples {1} --noise {add_noise} --batch {add_batch} --intervention FALSE")
            )
        geff =  pd.read_csv(cache_path/"R_out"/"geff.csv", index_col=0).to_numpy()
        region_to_gene = pd.read_csv(cache_path/"R_out"/"region_to_gene.csv", index_col=0).to_numpy()
        region_to_tf = pd.read_csv(cache_path/"R_out"/"region_to_tf.csv", index_col=0).to_numpy()

    if sem_regime == "pert_grn":
        # generate perturbed data
        for n, context in enumerate(gt_interv.T[1:].astype(bool)):
            pert_grn = beta.copy()
            if pert_type == "dCas9":
                pert_grn[: , context] *= pert_strength
            if pert_type == "Cas9":
                pert_grn[context, :] *= pert_strength

            format_for_scMultiSim(pert_grn, cache_path, pseudocounts)
            print(os.system(
                f"Rscript {scMultiSim.name} --n_genes {n_genes} --cache {cache_path.as_posix()} --n_samples {n_samples_per_pert} --noise {add_noise} --batch {add_batch} --intervention FALSE")
                )
            if n_samples_control == 0 and n == 0:
                # add header if necessary
                df = pd.read_csv(cache_path/"R_out"/"counts.csv", index_col=0).T
                #if normalize:
                #    df = normalize_data(df)
                df.to_csv(cache_path/"rna_matrix.csv")
                if create_mask:
                    df = pd.read_csv(cache_path/"R_out"/"atac_counts.csv", index_col=0).T
                    df.to_csv(cache_path/"atac_matrix.csv")
            else:
                df = pd.read_csv(cache_path/"R_out"/"counts.csv", index_col=0).T
                #if normalize:
                #    df = normalize_data(df)
                df.to_csv(cache_path/"rna_matrix.csv",mode="a", header=False)
            if create_mask and n < len(train_gene_ko):
                # remove unregulated regions and genes, added by scMultiSim
                #region_to_gene = region_to_gene[:,:-n_unregulated]
                #unregulated_regions = np.array([np.sum(region_to_gene, axis=1) == 0, np.sum(region_to_tf, axis=1) == 0])
                #unregulated_regions = np.min(unregulated_regions, axis=0)
                #region_to_tf = region_to_tf[~unregulated_regions]
                #region_to_gene = region_to_gene[~unregulated_regions]
                if region_to_tf.shape[1] < n_genes:
                    pad_shape = (region_to_tf.shape[0], n_genes-region_to_tf.shape[1])
                    region_to_tf = np.concat([region_to_tf, np.zeros(pad_shape)], axis=1)

                df = pd.read_csv(cache_path/"R_out"/"atac_counts.csv", index_col=0).T
                #df = df.loc[:,~unregulated_regions]
                df.to_csv(cache_path/"atac_matrix.csv", mode="a", header=False)

    # format data for use by create_loaders
    full_rna = pd.read_csv(cache_path/"rna_matrix.csv", index_col=0)
    full_rna = full_rna.reset_index(drop=True).to_numpy()
    if normalize:
        full_rna = normalize_data(full_rna)

    mask = None
    if create_mask:
        full_atac = pd.read_csv(cache_path/"atac_matrix.csv", index_col=0)
        full_atac = full_atac.reset_index(drop=True).to_numpy()
        if normalize:
            full_atac = normalize_data(full_atac, scale=False)

        mask = get_mask2(
            atac=full_atac[(sim_regime == 0)[:len(full_atac)]].T,
            region_to_gene = region_to_gene,
            region_to_tf = region_to_tf,
            **mask_kwargs
        )
        y_true = geff > 0.001

        y_true = y_true.flatten()
        mask_eval = expit((mask-np.mean(mask))/np.std(mask))

        print("average precision vs geff: ",average_precision_score(y_true, mask_eval.flatten()))

        if mask.shape != beta.shape:
            for n, (m, b) in enumerate(zip(mask.shape, beta.shape)):
                if m < b:
                    pad = [0, 0]
                    pad[n] = b-m
                    mask = np.pad(mask, pad, mode = "minimum")
                    print("mask is smaller than grn! Consider using pseudocounts.")
                if m > b:
                    print("mask is bigger than grn! Check RNA trimming was correct.")
                    if n:
                        mask = mask[:,:b-m]
                    else:
                        mask = mask[:b-m]
                    full_rna = full_rna[:,:b-m]

        if verbose and not generate_graph:
            with open(cache_path/"generation_kwargs.json", "r") as rf:
                dist = json.load(rf)
            fig = plot_comparison(mask, beta, dist)
            fig.savefig(cache_path/"beta_vs_mask.pdf")
            fig = plot_comparison(mask, geff, dist)
            fig.savefig(cache_path/"geff_vs_mask.pdf")
            fig.show()

        #evaluate the mask
        #if mask.shape != geff.shape:
        #    print("Padding mask!")
        #    mask = np.pad(mask, [geff.shape[n] - mask.shape[n] for n in range(len(mask.shape))], mode = "minimum")
        y_true = beta > 0.001


        y_true = y_true.flatten()
        mask_eval = expit((mask-np.mean(mask))/np.std(mask))
        precision = average_precision_score(y_true, mask_eval.flatten())
        print("average precision vs beta: ",precision)


    if verbose:
        def st_hist(ax, X, label=None, color=None, **kwargs):
            return ax.hist(X, label=label, 
                           color=color,
                           density=True, bins=np.linspace(-3, 20, 100), histtype="stepfilled", **kwargs)

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:gray']

        print("Checking perturbation effects...")
        # test if change in expression is significant for each gene in each dataset for each condition against normal
        results = np.empty(len(contexts))
        grn = beta.T
        alpha = 0.05
        ci = list()
        print("na count:",np.isnan(full_rna).sum())
        non_perturbed = full_rna[sim_regime == 0].copy()
        print("non_perturbed mean",np.mean(non_perturbed))
        print("non_perturbed std",np.std(non_perturbed))
        # for each condition
        fig, ax = plt.subplots(ncols=(len(contexts)+1))
        ax = ax.flatten()
        genes = np.arange(n_genes)
        fig.suptitle("Perturbation effects")
        for n, context in enumerate(gt_interv.T[1:].astype(bool)):
            p_values = list()
            weights = np.sum(grn[context], axis=0)
            down_mask = weights>0.1
            weights = weights[down_mask]
            downstream_genes = genes[down_mask]
            if len(downstream_genes) == 0:
                continue
            st_hist(ax=ax[n],X=non_perturbed[:, downstream_genes].flatten(), label=f"unperturbed downstream genes", color = colors[-1],alpha=0.8)
            targeted = full_rna[sim_regime == n+1].copy()
            print("targeted mean", np.mean(targeted[:, downstream_genes]))
            print("targeted std", np.std(targeted[:, downstream_genes]))
            #print("targeted",np.sum(np.isnan(targeted), axis=1))
            #print(targeted.shape)
            st_hist(ax=ax[n], X=targeted[:, downstream_genes].flatten(), label=f"perturbed in {downstream_genes.tolist()}", color=colors[0],alpha=0.5)

                #p_values.append(ttest_rel(adata[~adata.obs["perturbed"], tf].X[:400],b = adata[adata.obs["target"] == tf, tf].X, alternative="greater").pvalue<alpha)
            for gene in downstream_genes:
                # unperturbed expression of gene
                a = non_perturbed[:,gene].copy()
                # in tf perturbed epression of gene
                b = targeted[:,gene].copy()
                #np.random.shuffle(a)
                t = ttest_rel(a[:b.shape[0]], b, alternative="two-sided")
                p = t.pvalue
                ci.append(np.array(t.confidence_interval()))
                p_values.append(p<alpha)
                
            ax[n].set_xlim(-3, 5)
            ax[n].set_title(contexts[n])
            p_values = np.array(p_values).astype(float)
            ##weights /= np.mean(weights)
            ##weights = expit(weights)+0.5

            #p_values = p_values*weights
            ax[n].plot([], [], " ", label=f"Proportion of significant\ndifference in expression {np.mean(p_values).round(3)}")
            print(f"Proportion of significant\ndifference in expression {np.mean(p_values).round(3)} for target: {contexts[n]} with downstream: {downstream_genes}")
            ax[n].legend(loc="upper right")
            results[n] = np.mean(p_values).round(3)
        ax[n+1].imshow(grn, aspect = "auto")
        fig.set_figwidth((len(contexts)+1)*2)
        fig.show()
        fig.savefig(cache_path/"pert_effects.pdf")

    
    if not verbose:
        # clear the cache dir
        os.system(f"rm -r {cache_path}")
    else:
        # clear R_out
        os.system(f"rm -r {cache_path/"R_out"}")



    return  precision, mask.T, full_rna, gt_interv, sim_regime, beta.T

n_genes = 10
grn_shape = (0,0)
grn_params = {"sparsity":0.8}
n_samples_control = 500
n_samples_per_pert = 250

train_gene_ko=list(np.arange(10).astype(str))
test_gene_ko = []
while len(test_gene_ko)< n_genes*2:
    test_gene_ko.append(tuple(np.random.choice(n_genes, size=2, replace=False).astype(str)))
    test_gene_ko=list(set(test_gene_ko))
test_gene_ko = [f"{x[0]},{x[1]}" for x in test_gene_ko]
print(f"test_gene_ko: {test_gene_ko}")

pert_type = "dCas9"
pert_strength = 0
add_noise = False
add_batch = False
cache_path = Path("./scMultiSim_cache/")
scMultiSim = Path("./scMS_arg.R")
create_mask = True
normalize = True
pseudocounts = True
generate_graph = False
verbose = True
mask_kwargs = {}
graph_kwargs ={}# {"fit_to_standard" : True,
        #"standard_grn_path" : "../old_data/run_04/geff.csv",
        #"sample_connections" : True}
#TODO:
# Replicate optimization.ipynb classification
# remove perturbed from mask
#

precision, mask, full_rna, gt_interv, sim_regime, beta = generate_data(
        n_genes = n_genes,
        grn_shape = grn_shape,
        grn_params = grn_params,
        n_samples_control = n_samples_control,
        n_samples_per_pert = n_samples_per_pert,
        train_gene_ko = train_gene_ko,
        test_gene_ko = test_gene_ko,
        pert_type = pert_type,
        pert_strength = pert_strength,
        add_noise = add_noise,
        add_batch = add_batch,
        cache_path = cache_path,
        scMultiSim = scMultiSim,
        create_mask = create_mask,
        normalize=normalize,
        pseudocounts = pseudocounts,
        mask_kwargs = mask_kwargs,
        generate_graph=generate_graph,
        graph_kwargs=graph_kwargs,
        verbose = verbose,
        sem_regime = "pert_cif"
)


raise NotImplementedError("End of debugging!")

def generate_grn(
        n_genes = 10,
        grn_shape = (0,0),
        grn_params = {},
        fit_to_standard = False,
        standard_grn_path = None,
        distribute_TFs = True,
        out_path = Path("../"),
        verbose = False,
        eigenvalues = True,
        clip_standard = None,
        sample_connections =False,
        psd = True
        ):
    """
    Function to sample a grn from a bimodal normal distribution.

    Args.:
        n_genes
        grn_params (dict): dict with loc1, std1, loc2, std2, sparsity.
        fit_to_standard (bool): Wether to simulate based on a prior grn.
            In this case, the other parameters are derived from the standard grn.
        standard_grn_path (str or Path): Path to a csv containing the standard grn.
        distribute_TFs (bool): Determines if TFs will be the first n_TFs genes
            or distributed randomly.
        verbose (bool)
        eigenvalues (bool): If True the eigenvalues of the grn will be positive.
        clip_standard (int): If not None, the last clip_standard genes
            of the standard will be clipped before processing.
        sample_connections (bool): Wether to sample from connectivity_distribution.
    Returns:
        new_grn (np.ndarray)
        generation_kwargs (dict)
    """
    def fit_to_grn(grn:np.ndarray):
        """Function to fit a bimodal Normal distribution to grn values."""
        values = grn.flatten()
        values = np.sort(values[values!=0])
        values1= values[:len(values)//2]
        values2= values[len(values)//2:]

        loc1, std1 = norm.fit(values1)
        loc2, std2= norm.fit(values2)
        return loc1, std1, loc2, std2

    if grn_shape == (0,0):
        grn_shape = (n_genes, n_genes)
        
    if verbose:
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)


    TF_names = np.arange(1, grn_shape[1]+1).astype(str)
    gene_names = np.arange(1, grn_shape[0]+1).astype(str)
    if distribute_TFs:
        TF_names = np.random.choice(np.arange(1, grn_shape[0]+1), size=grn_shape[1], replace=False).astype(str)
        gene_names = np.arange(1, grn_shape[0]+1).astype(str)
    if verbose:
        print(f"TFs are {TF_names.tolist()}")

    ## First we create sample a grn form a bimodal normal distribution
    # To get the parameters we can either use an input or we can model of of the standard_grn
    if fit_to_standard:
        standard_grn = pd.read_csv(standard_grn_path, index_col=0).to_numpy()
        if clip_standard!=None:
            standard_grn=standard_grn[:-clip_standard]
        print("Using supplied standard grn")
        loc1, std1, loc2, std2 = fit_to_grn(standard_grn)
        grn_sparsity = np.sum(standard_grn==0).sum()/np.prod(standard_grn.shape)
        sparsity = grn_sparsity
        print(f"Gt sparsity: {grn_sparsity}")
    else:
        print("Using supplied parameters")
        loc1 = grn_params.get("loc1", 2)
        std1 = grn_params.get("std1", 0.4)
        loc2 = grn_params.get("loc2", 3.8)
        std2 = grn_params.get("std2", 0.6)
        sparsity = grn_params.get("sparsity", 0.8)
        standard_grn = None
    if fit_to_standard and sample_connections:
            # also fit connectivity distribution (TFs per gene)
            con = np.sum(standard_grn>0, axis=1)/standard_grn.shape[1]
            val, counts = np.unique(con, return_counts=True)
            counts = counts / np.sum(counts)
    
    success = False
    while not success:
        if fit_to_standard and sample_connections:
            connectivities = np.random.choice(val, size=grn_shape[0], replace=True, p=counts)
            connectivities *= grn_shape[1]
            connectivities = np.round(connectivities).astype(int)
        
            form = np.zeros(grn_shape)
            for n, con in enumerate(connectivities):
                row = np.append(np.ones(con), np.zeros(grn_shape[1]-con))
                np.random.shuffle(row)
                form[n] = row

        else:
            form = np.random.rand(*grn_shape)>sparsity
        
        if grn_shape[0] == grn_shape[1]:
            form = form *(np.ones(form.shape)-np.identity(n_genes))
        form = form.astype(bool)
        # check if all genes are involved
        for n in range(n_genes):
            while np.sum(np.append(form[n], form[:,n])) == 0:
                form[:,n] = np.random.rand(n_genes)>sparsity
                form[n] = np.random.rand(n_genes)>sparsity
                form[n,n] = 0


        if np.sum(form) % 2 == 0:
            success = True
        else:
            if verbose:
                print("GRN connectivities not multiple of 2")
            success=False
            continue
        form_sparsity = np.sum(form==0).sum()/np.prod(form.shape)
        if np.abs(form_sparsity-sparsity) > 0.05:
            if verbose:
                print(f"GRN sparsity {form_sparsity} too high, resampling!")
            success=False
            continue
        n = np.sum(form).astype(int)
        # fill the connections with samples from the bimodal dist.
        sampled1 = np.random.normal(loc=loc1, scale=std1, size=n//2)
        sampled2 = np.random.normal(loc=loc2, scale=std2, size=n//2)
        sampled = np.append(sampled1, sampled2)

        np.random.shuffle(sampled)
        values = np.zeros(grn_shape)
        values[form] = np.absolute(sampled)
        success = True
        
        #if psd:
        #    B = np.identity(values.shape[0])-values
        #    #B= B + np.exp(np.identity(B.shape[0])*B)
        #    B= B + np.diag(np.exp(np.diag(B)))
        #    print(np.exp(np.identity(B.shape[0])*B))
        #    print(np.identity(B.shape[0]))
        #    #m = LowerCholeskyTransform()(torch.tensor(B)).numpy()
        #    ##m = np.sqrt(np.abs(m))
        #    #zero = m == 0
        #    ##m[~zero] = m[~zero] - (np.max(m)-1)
        #    ##B = LowerCholeskyTransform().inv(torch.tensor(m)).numpy()
        #    #B = B@B.T
        #    #B = B - np.full(B.shape, np.max(B)-1)
        #    #B = B+B.T
        #    print(B)
        #    values = np.identity(B.shape[0])-B
        ## normalize and scale
        #zero = values ==0 
        #values = np.abs(values)#/np.std(values)
        #values[zero] = 0
        if eigenvalues:
            #check if the eigenvalues of B are >0
            B = np.identity(values.shape[0])-values
            eig_values = np.linalg.eigvals(B)
            print(f"Eigenvalues of B:\n{eig_values}")
            if not (eig_values>0).all():
                if verbose:
                    print("Eigenvalues not >0. Repeating...")
                success = False

    
    new_grn_sparsity = np.sum(values==0).sum()/np.prod(values.shape)
    print("GRN sparsity: ",new_grn_sparsity)

    if verbose:
        generation_kwargs = {
            "loc1":loc1, 
            "std1":std1, 
            "loc2":loc2, 
            "std2":std2,
            "sparsity": sparsity,
            "n_genes":n_genes,
        }
        fig = plot_comparison(values, standard_grn, dist = generation_kwargs)
        fig.savefig(out_path/f"{grn_shape[0]}gene{grn_shape[1]}tfGRN.pdf")
        fig.show()

        new_grn=pd.DataFrame(values, columns=TF_names, index=gene_names)
        new_grn.to_csv(out_path/f"{grn_shape[0]}gene{grn_shape[1]}tfGRN.csv")

        if sample_connections:
            generation_kwargs["connectivity_dist"]=(val.tolist(), counts.tolist())
        with open(out_path/"generation_kwargs.json", "w") as wf:
            json.dump(generation_kwargs, fp=wf)

    return values



# Input values for grn simulation
standard_grn = False
pert_strength = 0.01
samples_per_pert = 400

shape=(10, 10)
old_data_path = Path("../../bicycle_data/data_run000")
grn = pd.read_csv(old_data_path/"10gene10tfGRN.csv", index_col=0)
grn_params = [2, 0.4, 3.8, 0.6, 0.8]

distribute_TFs = False

out_path = Path("../")
run_id = get_id(dir_path=out_path, id_length=3, prefix="data_run")
out_path = out_path/run_id
print(f"Run id: {run_id}\n Output path: {out_path}")
if not out_path.exists():
    out_path.mkdir(parents=True, exist_ok=True)
cache_path = Path("./")/run_id
cache_path.mkdir(parents=False, exist_ok=False)


if standard_grn:
    fig,_,_,_,_ = fit_to_grn(grn.to_numpy())
    fig.savefig(out_path/"standard_grn.pdf")
    shape= grn.shape
    values = grn.to_numpy().astype(float)

TF_names = np.arange(1, shape[1]+1).astype(str)
gene_names = np.arange(1, shape[0]+1).astype(str)
if distribute_TFs:
    TF_names = np.random.choice(np.arange(1, shape[0]+1), size=shape[1], replace=False).astype(str)
    gene_names = np.arange(1, shape[0]+1).astype(str)
print(f"TFs are {TF_names.tolist()}")

## First we create sample a grn form a bimodal normal distribution
# To get the parameters we can either use an input or we can model of of another grn
if not standard_grn:
    if not grn is None:
        print("Using supplied grn")
        fig, loc1, std1, loc2, std2 = fit_to_grn(grn.to_numpy())
        sparsity = np.sum(grn==0).sum()/np.prod(grn.shape)
        print(f"Gt sparsity: {sparsity}")
        plt.clf()

    else:
        print("Using supplied parameters")
        loc1, std1, loc2, std2, sparsity = grn_params

    
    # also fit connectivity distribution
    con = np.sum(grn>0, axis=1)/grn.shape[1]
    val, counts = np.unique(con, return_counts=True)
    counts = counts / np.sum(counts)
    success = False
    while not success:
        connectivities = np.random.choice(val, size=shape[0], replace=True, p=counts)
        connectivities *= shape[1]
        connectivities =np.round(connectivities).astype(int)
        if np.sum(connectivities, axis = None) % 2 == 0:
            success = True
    form = np.empty(shape)
    for n, con in enumerate(connectivities):
        row = np.append(np.ones(con), np.zeros(shape[1]-con))
        np.random.shuffle(row)
        form[n] = row
    n = np.sum(form>0, axis=None)
    values1 = np.random.normal(loc=loc1, scale=std1, size=n//2)
    values2 = np.random.normal(loc=loc2, scale=std2, size=n//2)
    values = np.append(values1, values2)
    np.random.shuffle(values)
    form[form>0] = values[values>0]
    values = form
print("GRN sparsity: ",np.sum(values==0).sum()/np.prod(values.shape))

if not standard_grn:
    plt.rcParams["figure.figsize"] = (10, 10)
    if not grn is None:
        plt.subplot(1,4,1)
        plt.imshow(grn)
        plt.title("Original grn")

    plt.subplot(1,4,4)
    plt.imshow(values)
    plt.title("Modeled grn")
    plt.subplot(1,4,(2,3))
    plt.hist(values[values!=0].flatten(), bins=20, color="green", density=True, alpha = 0.6, label="Modeled distribution")
    if not grn is None:
        plt.hist(grn[grn!=0].to_numpy().flatten(), bins=20, color="red", density=True, alpha = 0.3, label="Original grn")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    pdf1 = norm.pdf(x, loc1, std1)
    pdf2 = norm.pdf(x, loc2, std2)
    plt.plot(x, pdf1/2, label=f"pdf1 with\nloc:{loc1}\nstd:{std1}", color = "orange", linewidth = 2)
    plt.plot(x, pdf2/2, label=f"pdf2 with\nloc:{loc2}\nstd:{std2}", color = "orange", linewidth = 2)

    plt.ylabel("Density")
    plt.xlabel("Interaction capacity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/data/toulouse/bicycle/notebooks/experiments/masking/presentation/{shape[0]}gene{shape[1]}tfGRN.pdf")
    plt.savefig(out_path/f"{shape[0]}gene{shape[1]}tfGRN.pdf")


# add pseudocounts
#values += 0.000001

# convert the grn to a for scMultiSim interpretable form and save
new_grn=pd.DataFrame(values, columns=TF_names, index=gene_names)
params= np.empty((np.prod(new_grn.shape), 3))
n=0
for gene,row in new_grn.iterrows():
    for tf, effect in row.items():
        params[n] = [gene, tf, effect]
        n+=1

np.random.shuffle(params)
params = pd.DataFrame(params, columns=["regulated.gene","regulator.gene","regulator.effect"])
params=params.loc[params["regulator.effect"]!=0]
params = params[~params.drop(columns="regulator.effect").duplicated()]

# formatting for R to understand it
params["regulated.gene"] = params["regulated.gene"].astype(int)
params["regulator.gene"] = params["regulator.gene"].astype(int)
params.index = np.arange(1, len(params)+1).astype(str)
new_grn.to_csv(out_path/f"{shape[0]}gene{shape[1]}tfGRN.csv")
params.to_csv(cache_path/f"food_for_R.csv", quoting=csv.QUOTE_NONNUMERIC)

print("Simulating unperturbed data")
# Feed the data to R
print(os.system(f"Rscript scMS.R {cache_path}"))

# read and convert to adata
df = pd.read_csv(cache_path/"R_out"/"counts.csv", index_col=0).T
df.to_csv(cache_path/"rna_matrix.csv",mode="a")
df = pd.read_csv(cache_path/"R_out"/"atac_counts.csv", index_col=0).T
df.to_csv(cache_path/"atac_matrix.csv",mode="a")
# copy data to a save place
(out_path/"unperturbed_data").mkdir(parents=True, exist_ok=True)

os.system(f"cp {cache_path/'R_out/*'} {out_path/'unperturbed_data/'}")

# perturb each tf
for tf in new_grn.columns:
    pert_new_grn = new_grn.copy()
    pert_new_grn[tf] = pert_new_grn[tf]*pert_strength
    params= np.empty((np.prod(pert_new_grn.shape), 3))
    n=0
    for gene,row in pert_new_grn.iterrows():
        for tf_n, effect in row.items():
            params[n] = [gene, tf_n, effect]
            n+=1
    np.random.shuffle(params)
    params = pd.DataFrame(params, columns=["regulated.gene","regulator.gene","regulator.effect"])
    params["regulated.gene"] = params["regulated.gene"].astype(int)
    params["regulator.gene"] = params["regulator.gene"].astype(int)
    params=params.loc[params["regulator.effect"]!=0]
    params = params[~params.drop(columns="regulator.effect").duplicated()]

    params.index = np.arange(1, len(params)+1).astype(str)
    params.to_csv(cache_path/f"food_for_R.csv", quoting=csv.QUOTE_NONNUMERIC)
    
    print(f"Simulating perturbed in {tf}")
    # Feed the data to R
    print(os.system(f"Rscript scMS_pert.R {cache_path}"))
    # read and convert to adata
    df = pd.read_csv(cache_path/"R_out"/"counts.csv", index_col=0).T
    df.to_csv(cache_path/"rna_matrix.csv",mode="a", header=False)
    df = pd.read_csv(cache_path/"R_out"/"atac_counts.csv", index_col=0).T
    df.to_csv(cache_path/"atac_matrix.csv",mode="a", header=False)
    # copy data to a save place

    (out_path/"perturbed_data"/tf).mkdir(parents=True, exist_ok=True)
    os.system(f"cp {cache_path/'R_out/*'} {out_path/'perturbed_data'/tf}/")

# process RNA data
df = pd.read_csv(cache_path/"rna_matrix.csv", index_col=0)
adata = sc.AnnData(df)
adata.var_names = df.columns.to_numpy(dtype=str)
adata.var_names_make_unique(join="-")
adata.obs_names_make_unique("-")
n_samples_unpert = len(df)-samples_per_pert*len(TF_names)
# "perturbed" for format_data function in evaluate
adata.obs["perturbed"] = [False if n<n_samples_unpert else True for n in range(len(df))]

targets = np.repeat(TF_names,samples_per_pert)

# "target_genes" for format_data function in evaluate
adata.obs["target"] = [np.nan if n<n_samples_unpert else targets[(n-len(df))] for n in range(len(df))]

# write full adata for full bicycle run
adata.obs["target_genes"] = ["" if n<n_samples_unpert else targets[(n-len(df))] for n in range(len(df))]
adata.write_h5ad(out_path/"ready_full_rna.h5ad")
# save as csv for mask analysis
df = pd.DataFrame(adata.X)
df.to_csv(out_path/"processed_rna.csv")


atac_base = pd.read_csv(cache_path/"atac_matrix.csv", index_col=0)
adata = sc.AnnData(atac_base)
adata.var_names = atac_base.columns.to_numpy(dtype=str)
adata.obs_names_make_unique("-")
adata.obs["perturbed"] = [False if n<n_samples_unpert else True for n in range(len(df))]

adata.obs["target"] = [np.nan if n<n_samples_unpert else targets[(n-len(df))] for n in range(len(df))]
# write full adata for full bicycle run
adata.obs["target_genes"] = ["" if n<n_samples_unpert else targets[(n-len(df))] for n in range(len(df))]
adata.write_h5ad(out_path/"ready_full_atac.h5ad")
df = pd.DataFrame(adata.X)
df.to_csv(out_path/"processed_atac.csv")

# clean environment
os.system(f"rm -r {cache_path}")
