
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
from bicycle.utils.mask_utils import get_mask2, normalize_data
from sklearn.metrics import average_precision_score
import torch
import json
from scipy.stats import ttest_rel

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
        psd = True,
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
            psd=psd,
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
        # Empty string represents control
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

        df = pd.read_csv(cache_path/"R_out"/"counts.csv", index_col=0).T
        
        df.to_csv(cache_path/"rna_matrix.csv")

        # generate geff and region matrices
        geff =  pd.read_csv(cache_path/"R_out"/"geff.csv", index_col=0).to_numpy()
        region_to_gene = pd.read_csv(cache_path/"R_out"/"region_to_gene.csv", index_col=0).to_numpy()
        region_to_tf = pd.read_csv(cache_path/"R_out"/"region_to_tf.csv", index_col=0).to_numpy()

        if create_mask:
            # get atac data for masking
            df = pd.read_csv(cache_path/"R_out"/"atac_counts.csv", index_col=0).T
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
                df.to_csv(cache_path/"rna_matrix.csv",mode="a", header=False)
            if create_mask and n < len(train_gene_ko):
                if region_to_tf.shape[1] < n_genes:
                    pad_shape = (region_to_tf.shape[0], n_genes-region_to_tf.shape[1])
                    region_to_tf = np.concat([region_to_tf, np.zeros(pad_shape)], axis=1)

                df = pd.read_csv(cache_path/"R_out"/"atac_counts.csv", index_col=0).T
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
            downstream_genes = genes[down_mask]
            if len(downstream_genes) == 0:
                continue
            st_hist(ax=ax[n],X=non_perturbed[:, downstream_genes].flatten(), label=f"unperturbed downstream genes", color = colors[-1],alpha=0.8)
            targeted = full_rna[sim_regime == n+1].copy()
            print("targeted mean", np.mean(targeted[:, downstream_genes]))
            print("targeted std", np.std(targeted[:, downstream_genes]))

            st_hist(ax=ax[n], X=targeted[:, downstream_genes].flatten(), label=f"perturbed in {downstream_genes.tolist()}", color=colors[0],alpha=0.5)

            for gene in downstream_genes:
                # unperturbed expression of gene
                a = non_perturbed[:,gene].copy()
                # in tf perturbed epression of gene
                b = targeted[:,gene].copy()
                t = ttest_rel(a[:b.shape[0]], b, alternative="two-sided")
                p = t.pvalue
                ci.append(np.array(t.confidence_interval()))
                p_values.append(p<alpha)
                
            ax[n].set_xlim(-3, 5)
            ax[n].set_title(contexts[n])
            p_values = np.array(p_values).astype(float)
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


