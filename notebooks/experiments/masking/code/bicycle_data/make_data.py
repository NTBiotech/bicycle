
import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from bicycle.utils.general import get_id
import scanpy as sc
import json

def fit_to_grn(grn:np.ndarray):
    values = grn.flatten()
    values = np.sort(values[values!=0])
    values1= values[:len(values)//2]
    values2= values[len(values)//2:]

    loc1, std1 = norm.fit(values1)
    loc2, std2= norm.fit(values2)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    ax[0].imshow(grn)
    ax[1].hist(np.random.normal(loc=loc1, scale=std1, size=len(values)*10), bins=25, alpha=0.2, color="red", label="Sampled values 1")
    ax[1].hist(np.random.normal(loc=loc2, scale=std2, size=len(values)*10), bins=25, alpha=0.2, color="blue", label="Sampled values 2")
    ax[1].hist(values, bins=20, color="green", alpha = 0.6, label="Actual distribution")
    xmin, xmax = ax[1].get_xlim()
    x = np.linspace(xmin, xmax, 200)
    pdf1 = norm.pdf(x, loc1, std1)
    pdf2 = norm.pdf(x, loc2, std2)
    ax[1].plot(x, pdf1/2, label=f"pdf1 with\nloc:{loc1}\nstd:{std1}", color = "orange", linewidth = 2)
    ax[1].plot(x, pdf2/2, label=f"pdf2 with\nloc:{loc2}\nstd:{std2}", color = "orange", linewidth = 2)
    ax[1].legend()
    return fig, loc1, std1, loc2, std2

def generate_grn(
        n_genes = 10,
        grn_sim_params = [],
        sparsity = 0.8,
        fit_to_standard = False,
        standard_grn_path = None,
        distribute_TFs = True,
        out_path = Path("../"),
        verbose = False,
        eigenvalues = True,
        clip_standard = None,
        sample_connections =False,
        ):
    """
    Function to sample a grn from a bimodal normal distribution.

    Args.:
        grn_shape (tuple): Dimensions of the grn.
        grn_sparsity (float): Proportion of non-zero values in the grn.
        grn_sim_params (list): List of tuples with [(loc1, std1), (loc2, std2)].
        connectity_dist (tuple): ([Number of connections], [Probability of Number of connections])
            per gene.
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
    grn_shape = (n_genes, n_genes)
    run_id = get_id(dir_path=out_path, id_length=3, prefix="data_run")
    out_path = out_path/run_id
    print(f"Run id: {run_id}\n Output path: {out_path}")

    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)


    TF_names = np.arange(1, grn_shape[1]+1).astype(str)
    gene_names = np.arange(1, grn_shape[0]+1).astype(str)
    if distribute_TFs:
        TF_names = np.random.choice(np.arange(1, grn_shape[0]+1), size=grn_shape[1], replace=False).astype(str)
        gene_names = np.arange(1, grn_shape[0]+1).astype(str)
    print(f"TFs are {TF_names.tolist()}")

    ## First we create sample a grn form a bimodal normal distribution
    # To get the parameters we can either use an input or we can model of of the standard_grn
    if fit_to_standard:
        standard_grn = pd.read_csv(standard_grn_path, index_col=0)
        if clip_standard!=None:
            standard_grn=standard_grn[:-clip_standard]
        print("Using supplied standard grn")
        fig, loc1, std1, loc2, std2 = fit_to_grn(standard_grn.to_numpy())
        grn_sparsity = np.sum(standard_grn==0).sum()/np.prod(standard_grn.shape)
        print(f"Gt sparsity: {grn_sparsity}")
        if verbose:
            fig.show()
        plt.clf()

    else:
        print("Using supplied parameters")
        (loc1, std1), (loc2, std2) = grn_sim_params
    if fit_to_standard:
        if sample_connections:
            # also fit connectivity distribution (TFs per gene)
            con = np.sum(standard_grn>0, axis=1)/standard_grn.shape[1]
            val, counts = np.unique(con, return_counts=True)
            counts = counts / np.sum(counts)
        else:
            sparsity = grn_sparsity
    
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
        
        form = form *(np.ones(form.shape)-np.identity(n_genes))
        form = form.astype(bool)
        # check if all genes are involved
        for n in range(n_genes):
            while np.sum(np.append(form[n], form[:,n])) == 0:
                form[:,n] = np.random.rand(n_genes)>sparsity
                form[n] = np.random.rand(n_genes)>sparsity
                form[n,n] = 0


        if verbose:
            print(f"Connectivities:\n{form}")
        if np.sum(form) % 2 == 0:
            success = True
        else:
            if verbose:
                print("GRN connectivities not multiple of 2")
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
        
        if eigenvalues:
            #check if the eigenvalues of B are >0
            B = np.identity(values.shape[0])-values
            eig_values = np.linalg.eigvals(B)
            if verbose:
                print(f"B:\n{B}")
                print(f"Eigenvalues:\n{eig_values}")
            if not (eig_values>0).all():
                if verbose:
                    print("Eigenvalues not >0. Repeating...")
                success = False

    new_grn_sparsity = np.sum(values==0).sum()/np.prod(values.shape)
    print("GRN sparsity: ",new_grn_sparsity)

    plt.rcParams["figure.figsize"] = (10, 10)
    if fit_to_standard:
        plt.subplot(1,4,1)
        plt.imshow(standard_grn)
        plt.title("Standard grn")
    plt.subplot(1,4,4)
    plt.imshow(values)
    plt.title("Modeled grn")
    plt.subplot(1,4,(2,3))
    plt.hist(values[values!=0].flatten(), bins=20, color="green", density=True, alpha = 0.6, label="Modeled distribution")
    if fit_to_standard:
        plt.hist(standard_grn[standard_grn!=0].to_numpy().flatten(), bins=20, color="red", density=True, alpha = 0.3, label="Original grn")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    pdf1 = norm.pdf(x, loc1, std1)
    pdf2 = norm.pdf(x, loc2, std2)
    plt.plot(x, pdf1/2, label=f"pdf1 with\nloc:{loc1}\nstd:{std1}", color = "orange", linewidth = 2)
    plt.plot(x, pdf2/2, label=f"pdf2 with\nloc:{loc2}\nstd:{std2}", color = "blue", linewidth = 2)
    plt.ylabel("Density")
    plt.xlabel("Interaction capacity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/data/toulouse/bicycle/notebooks/experiments/masking/presentation/{grn_shape[0]}gene{grn_shape[1]}tfGRN.pdf")
    plt.savefig(out_path/f"{grn_shape[0]}gene{grn_shape[1]}tfGRN.pdf")
    if verbose:
        plt.show()

    new_grn=pd.DataFrame(values, columns=TF_names, index=gene_names)
    new_grn.to_csv(out_path/f"{grn_shape[0]}gene{grn_shape[1]}tfGRN.csv")

    generation_kwargs = {
        "loc1":loc1, 
        "std1":std1, 
        "loc2":loc2, 
        "std2":std2,
        "sparsity": new_grn_sparsity,
        "n_genes":n_genes,
    }
    if sample_connections:
        generation_kwargs["connectivity_dist"]=(val.tolist(), counts.tolist())
    with open(out_path/"generation_kwargs.json", "w") as wf:
        json.dump(generation_kwargs, fp=wf)

    return new_grn.to_numpy(), generation_kwargs

#
os.chdir(Path(__file__).parent)
#
#scMS_path = Path("../scMultiSim_data/old_data/run_04")
#standard_grn_path = scMS_path/"geff.csv"
#grn = pd.read_csv(standard_grn_path, index_col=0)[:-10]
lengths = [8,10]
for n in lengths:
    new_grn, params = generate_grn(
        n_genes = n,
        grn_sim_params=[(2,0.4),(3.8,0.6)],
        sparsity=0.7,
        distribute_TFs=False,
        eigenvalues=True,
        verbose= False,
        out_path=Path("./")
        )

