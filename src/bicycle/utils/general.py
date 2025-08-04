def get_full_name(
    name_prefix,
    nlogo,
    seed,
    lr,
    n_genes,
    scale_l1,
    scale_kl,
    scale_spectral_loss,
    scale_lyapunov,
    gradient_clip_val,
    swa,
):
    return f"{name_prefix}_{nlogo}_{seed}_{lr}_{n_genes}_{scale_l1}_{scale_kl}_{scale_spectral_loss}_{scale_lyapunov}_{gradient_clip_val}_{swa}"

def get_custom_name(*args):
    """
    Function that converts all input variables into one string with values and one string with names.
    Format: "valueofv1_valueofv2_..._valueofvx". 
    Only works for variables that are registered in globals().

    """
    value_string = ""
    key_string = ""
    for key, value in globals().items():
        if value in args:
            value_string += f"{value}_"
            key_string += f"{key}_"


    return value_string[:-1], key_string[:-1]

from pathlib import Path
def get_id(dir_path: Path, id_length: int = 5, prefix: str = ""):
    """Generates a the next in dir_path not existing directory name id of length n out of ints. """
    
    id = None
    existing_ids = [d.name for d in dir_path.iterdir() if d.is_dir()]
    n = 0
    while id == None:
        id = prefix + f"0"*(id_length-len(str(n)))+ str(n)
        if id in existing_ids:
            id  = None
            n+=1
    
    return id
