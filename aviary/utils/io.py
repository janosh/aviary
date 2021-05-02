import os
from os.path import abspath, dirname


# absolute path to the project's root directory
ROOT = dirname(dirname(dirname(abspath(__file__))))


def bold(text):
    """Turn text string bold when printed.
    https://stackoverflow.com/q/8924173/4034025
    """
    return f"\033[1m{text}\033[0m"


def make_model_dir(model_name, ensemble=1, run_id="run_1"):
    """Generate the correct directories for saving model checkpoints,
    TensorBoard runs and other output depending on whether using ensembles
    or single models.
    """
    model_dir = f"{ROOT}/models/{model_name}"

    if ensemble > 1:
        for idx in range(ensemble):
            os.makedirs(f"{model_dir}/ens_{idx}", exist_ok=True)
    else:
        # bake run_id into the return value if in single-model mode
        if run_id:
            model_dir += f"/{run_id}"
        os.makedirs(model_dir, exist_ok=True)

    return model_dir
