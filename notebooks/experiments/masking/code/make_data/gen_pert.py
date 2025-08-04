"""Script for creating scRNA and scATAC perturbation data with scMultiSim"""

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
os.chdir(Path(__file__).parent)

pert_strength = 0.01
samples_per_pert = 5   # if changed also change in scMS_unseen.R
run_id = "data_run011"

out_path = Path("../")
out_path = out_path/run_id
if not out_path.is_dir():
    raise NotADirectoryError(f"Directory {out_path} not found!")

grn_path = ""
for p in out_path.iterdir():
    if p.name.endswith("GRN.csv"):
        grn_path = out_path / p.name
        break
if grn_path=="":
    raise FileNotFoundError("No *GRN.csv file found!")

grn = pd.read_csv(grn_path, index_col=0)

print(f"Run id: {run_id}\n Output path: {out_path}")


# enumerate all possible perturbation contexts
gene_idxs = np.arange(grn.shape[0])
contexts = np.zeros((1,grn.shape[0]), dtype=int)

for g in gene_idxs:
    new_contexts = contexts.copy()
    new_contexts[:,g] = 1
    contexts=np.concatenate([contexts, new_contexts])
# remove contexts <= 1
mask = np.sum(contexts, axis=1)>1
contexts = contexts[mask]
print(f"Shape of context vector{contexts.shape}")


# perturb each tf
cache_path = Path("./")/run_id
if cache_path.exists():
    os.system(f"rm -r {cache_path}")

cache_path.mkdir(parents=False, exist_ok=False)
for c in contexts:
    pert_new_grn = grn.copy()
    pert_new_grn.iloc[c] = pert_new_grn.iloc[c]*pert_strength
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

    # Feed the data to R
    print(os.system(f"Rscript scMS_unseen.R {cache_path}"))
    # read and convert to adata
    if (out_path/"unseen_rna_matrix.csv").exists():
        df = pd.read_csv(cache_path/"R_out"/"counts.csv", index_col=0).T
        df.to_csv(cache_path/"unseen_rna_matrix.csv",mode="a", header=False)
        df = pd.read_csv(cache_path/"R_out"/"atac_counts.csv", index_col=0).T
        df.to_csv(cache_path/"unseen_atac_matrix.csv",mode="a", header=False)
    else:
        df = pd.read_csv(cache_path/"R_out"/"counts.csv", index_col=0).T
        df.to_csv(cache_path/"unseen_rna_matrix.csv",mode="a")
        df = pd.read_csv(cache_path/"R_out"/"atac_counts.csv", index_col=0).T
        df.to_csv(cache_path/"unseen_atac_matrix.csv",mode="a")

target_genes = np.array([gene_idxs[c] for c in contexts])
np.save(out_path/"unseen_target_genes.npy", target_genes)

os.system(f"cp {cache_path/'unseen_rna_matrix.csv'} {out_path}/")
os.system(f"cp {cache_path/'unseen_atac_matrix.csv'} {out_path}/")

# clean environment
os.system(f"rm -r {cache_path}")
