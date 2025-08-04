import numpy as np
import pandas as pd
from pathlib import Path
from scMultiSim import generate_data
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, auc, precision_recall_curve
from bicycle.utils.mask_utils import get_mask2, get_sparsity, add_saltpepper, normalize_matrix, above_threshold, add_noise
from sklearn.decomposition import PCA
from scipy.special import expit
import os
os.chdir("/data/toulouse/bicycle/notebooks/experiments/masking")
def get_precision(grn, beta):
    grn=grn.flatten()
    beta=beta.flatten()
    # normalize to [0, 1]
    grn = (grn > 0).astype(int)
    #beta[beta<0] = 0
    beta = expit(beta)
    average_precision = average_precision_score(grn, beta)
    return average_precision
def get_auprc(grn, beta):
    grn=grn.flatten()
    beta=beta.flatten()
    # normalize to [0, 1]
    grn = (grn > 0).astype(int)
    #beta[beta<0] = 0
    beta = expit(beta)
    p,r,t = precision_recall_curve(grn, beta)
    auprc = auc(r,p)
    return auprc
def apply_pca(data_profile:dict, n_comp=None, mask=[]):
    new_profile = data_profile.copy()
    atac = new_profile["atac"]
    pca = PCA(svd_solver="full").fit(atac)
    atac=pca.transform(atac)
    if len(mask) >0:
        atac[:,mask] = 0
    if n_comp!= None:
        atac[:, n_comp:] = 0
    #print(np.min(atac, axis=None))
    #atac = atac + np.abs(np.min(atac, axis=None))
    atac = pca.inverse_transform(atac)
    new_profile["atac"]= atac
    return new_profile
def get_mask2(atac,
             region_to_gene,
             region_to_tf,
             threshold = False,
             percentile : int = 50,
             correlation = False,
             corr_normalize = False,
             corr_threshold = False,
             corr_threshold_percentile: int = 50,
             pseudocounts = False,
             ):
    """
    Args:
    atac (np.array): regions x samples
    params (dict): function parameters
    """
    if pseudocounts:
        atac += np.min(atac)*0.0001
    if correlation and atac.shape[1]>1:
        corr_atac = np.abs(np.corrcoef(atac))
        #plt.imshow(corr_atac)
        if corr_normalize:
#            print("norm")
            corr_atac = normalize_matrix(corr_atac)
        if corr_threshold:
#            print("threshold")
            corr_atac = above_threshold(corr_atac, corr_threshold_percentile)
    else:
        corr_atac = atac @ atac.T
    mask = region_to_gene.T @ corr_atac @ region_to_tf
    if threshold:
        mask = above_threshold(mask, percentile)
    return mask

def output_precision(mask_output):
    return mask_output[0]
def selection_wrap(
    mask_threshold,
    mask_percentile,
    mask_correlation,
    mask_corr_normalize,
    mask_corr_threshold,
    mask_corr_threshold_percentile,
    mask_pseudocounts,
    **config
    ):
    mask_kwargs={
        "threshold":mask_threshold,
        "percentile":mask_percentile,
        "correlation":mask_correlation,
        "corr_normalize":mask_corr_normalize,
        "corr_threshold":mask_corr_threshold,
        "corr_threshold_percentile":mask_corr_threshold_percentile,
        "pseudocounts":mask_pseudocounts,}
    return generate_data(**config, mask_kwargs=mask_kwargs)

def selection_wrap_pca(**input):
    return generate_data(
        n_genes = input.pop("n_genes"),
        grn_params = input.pop("grn_params"),
        n_samples_control = input.pop("n_samples_control"),
        n_samples_per_pert = input.pop("n_samples_per_pert"),
        train_gene_ko = input.pop("train_gene_ko"),
        test_gene_ko = input.pop("test_gene_ko"),
        pert_type = input.pop("pert_type"),
        pert_strength = input.pop("pert_strength"),
        add_noise = input.pop("add_noise"),
        add_batch = input.pop("add_batch"),
        cache_path = input.pop("cache_path"),
        scMultiSim = input.pop("scMultiSim"),
        create_mask = input.pop("create_mask"),
        normalize = input.pop("normalize"),
        pseudocounts = input.pop("pseudocounts"),
        generate_graph = input.pop("generate_graph"),
        verbose = input.pop("verbose"),
        sem = input.pop("sem"),
        noise_p=input.pop("noise_p"),
        mask_kwargs={"pca":True, "pcs":list(input.values())})
# Forward selection
def for_select(data, _parameters, _baseline_prec = 0, eval = get_precision, mask_func = get_mask2, eval_kwargs = {},):
    if _baseline_prec == 0:

        mask_output = mask_func(**_parameters, **data)
        _baseline_prec = eval(mask_output, **eval_kwargs)
        #print(_baseline_prec)

    precisions = []
    for key, value in _parameters.items():

        if value:
            precisions.append(0)
            continue
        if not type(value) is bool:
            precisions.append(0)
            continue
        params = _parameters.copy()
        params[key] = True
        mask_output = mask_func(**params, **data)
        precisions.append(eval(mask_output, **eval_kwargs))
    
    #print(precisions)
    #print(_parameters)
    max_prec = np.max(precisions)
    
    if max_prec <= _baseline_prec:
        return _parameters, _baseline_prec
    
    argmax = np.argmax(precisions)
    maxkey = [n for n in _parameters.keys()][argmax]
    _parameters[maxkey] = True

    return for_select(data, _parameters, max_prec, eval=eval, mask_func=mask_func)

# Reverse selection
def rev_select(data, _parameters, _baseline_prec = 0, eval = get_precision, mask_func = get_mask2, eval_kwargs = {},):
    if _baseline_prec == 0:
        mask_output = mask_func(**_parameters,**data)
        _baseline_prec = eval(mask_output, **eval_kwargs)
        #print(_baseline_prec)
    precisions = []
    keys = []
    for key, value in _parameters.items():

        if not value:
            continue
        if not type(value) is bool:
            continue
        params = _parameters.copy()
        params[key] = False
        mask_output = mask_func(**data, **params)
        precisions.append(eval(mask_output, **eval_kwargs))
        keys.append(key)
    #print(f"precisions: {precisions}")
    #print(f"keys: {keys}")
    if len(precisions) ==0:
        #max_prec = eval(grn,mask_func(**data, **_parameters))
        return _parameters, _baseline_prec
    
    max_prec = np.max(precisions)
    
    if max_prec <= _baseline_prec:
        return _parameters, _baseline_prec
    
    argmax = np.argmax(precisions)
    maxkey = keys[argmax]
    _parameters[maxkey] = False
    return rev_select(data, _parameters, max_prec, eval=eval, mask_func=mask_func)

config = {
    "n_genes":10,
    "grn_params":{},
    "n_samples_control": 500,
    "n_samples_per_pert": 0,
    "train_gene_ko": [],
    "test_gene_ko": [],
    "pert_type":"dCas9",
    "pert_strength":0,
    "add_noise":False,
    "add_batch": False,
    "cache_path": Path("./scMultiSim_cache/"),
    "scMultiSim": Path("./scMS.R"),
    "create_mask":True,
    "normalize":False,
    "pseudocounts":  True,
    "generate_graph":  True,
    "verbose": False,
    "sem" :"pert_grn",
}

threshold = 80

rev_parameters = {
    "mask_threshold" : True,
    "mask_percentile" : threshold,
    "mask_correlation" : True,
    "mask_corr_normalize" : True,
    "mask_corr_threshold" : True,
    "mask_corr_threshold_percentile" : threshold,
    "mask_pseudocounts" : True,
    }

for_parameters = {
    "mask_threshold" : False,
    "mask_percentile" : threshold,
    "mask_correlation" : False,
    "mask_corr_normalize" : False,
    "mask_corr_threshold" : False,
    "mask_corr_threshold_percentile" : threshold,
    "mask_pseudocounts" : False,
    }
print(threshold)

# increase noise on the region matrices
repeats = 3
step = 0.1
max = 1.1
rev_results = np.empty((len(rev_parameters.values()), repeats))
for_results = np.empty((len(for_parameters.values()), repeats))
for r in range(repeats):
    print(f"r: {r}")
    base_prec = output_precision(selection_wrap(**config, **rev_parameters))
    params, max_prec= rev_select(data=config,_parameters=rev_parameters.copy(), eval=output_precision, mask_func=selection_wrap)
    #print(params)
    del params["mask_percentile"]
    del params["mask_corr_threshold_percentile"]
    #print(max_prec)
    #print(base_prec)
    #for_parameters["mask_corr_threshold_percentile"] = p
    #for_parameters["mask_percentile"] =p
    rev_results[:, r] = np.append([max_prec, base_prec], list(params.values()))
    base_prec = output_precision(selection_wrap(**config, **for_parameters))
    params, max_prec= for_select(data=config,_parameters=for_parameters.copy(), eval=output_precision, mask_func=selection_wrap)
    #print(params.values())
    del params["mask_percentile"]
    del params["mask_corr_threshold_percentile"]
    #print(max_prec)
    #print(base_prec)
    for_results[:, r] = np.append([max_prec, base_prec], list(params.values()))

np.savetxt("rev_results",rev_results, delimiter=",")
np.savetxt("for_results",for_results, delimiter=",")

n_comp=33
for_parameters = {str(n):False for n in range(33)}
rev_parameters = {str(n):True for n in range(33)}
# increase noise on the region matrices
repeats = 1
step = 0.25
max = 1.1
rev_results = np.empty((len(rev_parameters.values()), repeats))
for_results = np.empty((len(for_parameters.values()), repeats))
rev_results = np.empty((int(max/step), n_comp+2, repeats))
for_results = np.empty((int(max/step), n_comp+2, repeats))
for r in range(repeats):
    print(f"Repeat: {r}")
    for n, p in enumerate(np.arange(0,max, step)):
        config.update({"noise_p":p})
        base_prec = output_precision(selection_wrap_pca(**config, **rev_parameters))
        params, max_prec= rev_select(data=config,
                                     _parameters=rev_parameters.copy(),
                                     eval=output_precision,
                                     mask_func=selection_wrap_pca)        
        rev_results[n, :, r] = np.append([max_prec, base_prec], list(params.values()))

        config.update({"noise_p":p})
        base_prec = output_precision(selection_wrap_pca(**config, **for_parameters))
        params, max_prec= for_select(data=config,
                                     _parameters=for_parameters.copy(),
                                     eval=output_precision,
                                     mask_func=selection_wrap_pca)
        for_results[n, :, r] = np.append([max_prec, base_prec], list(params.values()))
np.savetxt("rev_results_pca",rev_results, delimiter=",")
np.savetxt("for_results_pca",for_results, delimiter=",")