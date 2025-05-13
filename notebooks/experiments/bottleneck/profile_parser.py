"""
Submodule to parse cProfiler output files.
"""


import pandas as pd
from pathlib import Path
import re
import argparse

ANALYSIS_PATH = Path("/data/toulouse/bicycle/notebooks/experiments/bottleneck/data/analysis")
MODELS_PATH = Path("/data/toulouse/bicycle/notebooks/experiments/bottleneck/data/models")
PLOTS_PATH = Path("/data/toulouse/bicycle/notebooks/experiments/bottleneck/data/plots")
exclude=["test_run_00013", "test_run_00014"]

argparser = argparse.ArgumentParser()
argparser.add_argument("--MODELS_PATH", type=str)
argparser.add_argument("--PLOTS_PATH", type=str)
argparser.add_argument("--ANALYSIS_PATH", type=str)
argparser.add_argument("--exclude", type=list)
args=argparser.parse_args()
if args.MODELS_PATH:
    MODELS_PATH=args.MODELS_PATH

if args.PLOTS_PATH:
    PLOTS_PATH=args.PLOTS_PATH

if args.ANALYSIS_PATH:
    ANALYSIS_PATH=args.ANALYSIS_PATH

if args.exclude:
    exclude = args.exclude

def read_profile(path: Path, rank: bool = False):
    """
    Function to parse a 'FIT Profiler Report' from a textfile. 

    Returns:
    tuple: (
        df: pd.DataFrame containing all profiling data,
        df_sum: pd.DataFrame containing data for the function)
    """
    print(f"Parsing file: {path}")
    with open(path, "r") as rf:
        parsed_file = rf.read()
    parsed_file = parsed_file.split("\n\n\n")
    
    column_names = ["ncalls", "tot_time", "tot_percall", "cum_time", "cum_percall", "filename:lineno(function)", "Class", "Function", "Call_num", "Primitive_Call_num", "Time", "Rank", "Summary_index",]
    
    class_pattern = re.compile("\[\S+(?=\.)")
    function_pattern = re.compile("\w+(?=#)")
    callnum_pattern = re.compile("(?<=\s\s\s)\d+")
    primcallnum_pattern = re.compile("(?<=\()\d+")
    time_pattern= re.compile("\d+\.\d\d\d")
    rank_pattern = re.compile(":\s\d*\s")
    patterns = [class_pattern, function_pattern, callnum_pattern, primcallnum_pattern, time_pattern, rank_pattern]

    df_sum = pd.DataFrame(columns=column_names[6:-1])
    df = pd.DataFrame(columns=column_names)
    for n, section in enumerate(parsed_file):
        section = section.split("\n")
        title = "#".join(section[1:3])

        title_columns = [re.findall(pat, title) for pat in patterns]
        title_columns = [str(s[0]) if len(s)>0 else pd.NA for s in title_columns]
        df_sum.loc[len(df_sum)] = title_columns
        
        title_columns.append(len(df_sum)-1)
        for line in section[7:]:
            
            split = line.split(" ")
            while "" in split:
                split.remove("")
            if "/" in split[0]:
                split[0] = split[0].split("/")[0]

            df.loc[len(df), column_names[:5]] = split[:5]
            if re.findall("{.+}", line) != []:
                #print(re.findall("{.+}", line))
                df.iloc[len(df)-1, 5] = str(re.findall("{.+}", line)[0])
            else:
                #print("split", split[-1])
                df.iloc[len(df)-1, 5] = split[-1]

            df.iloc[len(df)-1, 6:] = title_columns
        
            
    return df, df_sum.iloc[:-1]

#read_profile("/data/toulouse/bicycle/notebooks/experiments/bottleneck/data/models/test_run_00030/profiler/fit-training_profile.txt")[1].iloc[-3:]
def filter_profiler_records(df, df_sum):
    print("Filtering...")
    df_sum=df_sum.astype({"Time": float,
                          "Call_num": int})
    df = df.astype({"tot_time": float,
                        "ncalls": int})
    clean_df = pd.DataFrame(columns=df.columns)
    for func_index, function in df_sum.Function.items():
        profiler_time = 0.0
        profiler_calls = 0
        sub_sel = df.loc[df["Function"] == function]
        for _, row in sub_sel.iterrows():
            if "profiler".casefold() in row["filename:lineno(function)"].casefold() or "posix".casefold() in row["filename:lineno(function)"].casefold():
                profiler_time += row["tot_time"]
                profiler_calls += row["ncalls"]
                continue
            else:
                clean_df.loc[len(clean_df)] = row
        df_sum.loc[func_index, "Time"] -= profiler_time
        df_sum.loc[func_index, "Call_num"] -= profiler_calls
    return clean_df, df_sum

def process_profile_data(df, df_sum):
    full_dtype_converter = {
    "ncalls": int,
    "tot_time": float,
    "tot_percall":float,
    "cum_time":float,
    "cum_percall":float,
    "filename:lineno(function)":str,
    "Class":str,
    "Function":str,
    "Call_num":int,
    "Primitive_Call_num":int,
    "Time":float,
    #"Rank":int,
    "Summary_index": int
    }
    dtype_converter = {
    "Class":str,
    "Function":str,
    "Call_num":int,
    "Primitive_Call_num":int,
    "Time":float,
    #"Rank":int,
    }
    df_sum["Primitive_Call_num"] = df_sum["Primitive_Call_num"].fillna(0)
    df_sum = df_sum.astype(dtype_converter)
    df_sum["Class_Function"] = df_sum["Class"] + "_" + df_sum["Function"]
    df_sum["is_callback"] = df_sum["Class"].apply(lambda x: "Callback" in x)

    df["Primitive_Call_num"] = df["Primitive_Call_num"].fillna(0)
    df = df.astype(full_dtype_converter)
    df["Class_Function"] = df["Class"] + "_" + df["Function"]
    df["filename_lineno(function)"] = df["filename:lineno(function)"]
    df = df.drop(columns=["filename:lineno(function)"])
    df["Class_Function_etc"] = df["Class_Function"] + ":" + df["filename_lineno(function)"]
    df["is_callback"] = df["Class"].apply(lambda x: "Callback" in x)

    return df, df_sum

# parse the models dict for profiler data from testrun_synthetic_benchmark.py
def fullparse_profiles(
        pretraining:bool = False,
        filtering:bool=True,
        pickle:bool=False,
        MODELS_PATH: Path = Path("/data/toulouse/bicycle/notebooks/experiments/bottleneck/data/models"),
        ANALYSIS_PATH:Path=Path("/data/toulouse/bicycle/notebooks/experiments/bottleneck/data/analysis"),
        hyperparameter_names: list = ["run_id","data_n_genes", "data_n_samples_control", "data_n_samples_per_perturbation", "batch_size", "n_epochs", "n_epochs_pretrain_latents", "scale_factor", "training_time", "pretraining_time"],
    ):
    """
    Standard run to parse all profile output files in MODELS_PATH inside subdirectories with name "test_run_<run_id>".

    Returns:
    tuple(
        hyperparameters: pd.DataFrame containing run hyperparameters for each id,
        profile_dict: dict() containing DataFrames for all training and pretraining runs
    )
    """

    profile_dict = dict()
    hyperparameters = pd.DataFrame(columns=hyperparameter_names)


    for subdir in MODELS_PATH.iterdir():
        if subdir.joinpath("profiler", "fit-training_profile.txt").exists():
            key = str(subdir.name)
            if key in exclude:
                print(f"Skipping {key}!")
                continue
            print(subdir)
        # get globals of run
            globs = pd.read_csv(PLOTS_PATH.joinpath(key, "globals.csv"), delimiter=",").set_index("0", drop=True).T
            available_paras = [n for n in hyperparameter_names if n in globs.columns]
            hyperparameters.loc[len(hyperparameters)] = globs[available_paras].iloc[1]

            df, df_sum = read_profile(subdir.joinpath("profiler", "fit-training_profile.txt"))

        # remove profiler traces
            if filtering:
                df, df_sum = filter_profiler_records(df, df_sum)

            df, df_sum = process_profile_data(df, df_sum)
            if pickle:
                if not ANALYSIS_PATH.joinpath(key).is_dir():
                    ANALYSIS_PATH.joinpath(key).mkdir(parents=True, exist_ok=True)
                df.to_pickle(ANALYSIS_PATH.joinpath(key, "full_training_profile.gz"))
                df_sum.to_pickle(ANALYSIS_PATH.joinpath(key, "training_profile.gz"))

            profile_dict[key] = dict()
            profile_dict[key]["full_training_profile"] = df
            profile_dict[key]["training_profile"] = df_sum

            # check for pretraining profiles
            if subdir.joinpath("profiler/fit-pretraining_profile.txt").exists() and pretraining:
                df, df_sum = read_profile(subdir.joinpath("profiler/fit-pretraining_profile.txt"))

                if filtering:
                    df, df_sum = filter_profiler_records(df, df_sum)

                df, df_sum = process_profile_data(df, df_sum)

                if pickle:
                    df.to_pickle(ANALYSIS_PATH.joinpath(key, "full_pretraining_profile.gz"))
                    df_sum.to_pickle(ANALYSIS_PATH.joinpath(key, "pretraining_profile.gz"))

                profile_dict[key]["full_pretraining_profile"] = df
                profile_dict[key]["pretraining_profile"] = df_sum
    if pickle:
        hyperparameters.to_csv(ANALYSIS_PATH.joinpath("parameters.csv"))

    return hyperparameters, profile_dict

fullparse_profiles(pickle=True)