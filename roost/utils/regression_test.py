from os.path import isfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from roost.core import Normalizer


def predict(model_class, test_set, checkpoint_path, device, robust, repeat):

    assert isfile(checkpoint_path), f"no checkpoint found at '{checkpoint_path}'"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    chk_robust = checkpoint["model_params"]["robust"]
    assert (
        chk_robust == robust
    ), f"checkpoint['robust'] != robust ({chk_robust} vs  {robust})"

    model = model_class(**checkpoint["model_params"], device=device)
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"])

    normalizer = Normalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])

    if "swa" in checkpoint.keys():
        model.swa = checkpoint["swa"]

        model_dict = model.swa["model_state_dict"]
        model.swa["model"] = AveragedModel(model)
        model.swa["model"].load_state_dict(model_dict)

    idx, comp, y_test, output = model.predict(test_set, repeat=repeat)

    df = pd.DataFrame({"idx": idx, "comp": comp, "y_test": y_test})

    output = output.cpu().squeeze()  # move preds to CPU in case model ran on GPU
    if robust:
        mean, log_std_al = [x.squeeze() for x in output.chunk(2, dim=1)]
        if repeat > 1:
            log_std_al = log_std_al.mean(-1)
            df["std_ep"] = (mean.std(-1) * normalizer.std).numpy()
            mean = mean.mean(-1)
        df["pred"] = normalizer.denorm(mean).numpy()
        df["std_al"] = (log_std_al.exp() * normalizer.std).numpy()
    else:
        if repeat > 1:
            df["std_ep"] = (output.std(-1) * normalizer.std).numpy() * 100
            output = output.mean(-1)
        df["pred"] = normalizer.denorm(output).numpy()

    return df


def regression_test(
    model_class,
    model_dir,
    ensemble_folds,
    test_set,
    data_params,
    robust,
    device="cpu",
    eval_type="checkpoint",
    repeat=1,  # use with MNF to get epistemic uncertainty
):
    """Evaluate an ensemble's performance on the test set"""

    print("\n------------ Evaluating model on test set ------------\n")

    test_set = DataLoader(test_set, **data_params)

    dfs, preds = [], []
    for ens in range(ensemble_folds):
        if ensemble_folds > 1:
            print(f"Evaluating Model {ens + 1}/{ensemble_folds}")
            checkpoint_path = f"{model_dir}/ens_{ens}/{eval_type}.pth.tar"
        else:
            checkpoint_path = f"{model_dir}/{eval_type}.pth.tar"

        df_i = predict(model_class, test_set, checkpoint_path, device, robust, repeat)

        preds.append(df_i.pred)

        col_map = {key: f"{key}_{ens}" for key in ["pred", "std_al", "std_ep"]}
        dfs.append(df_i.rename(columns=col_map))

    df = pd.concat(dfs, axis=1)  # combine all model outputs into single dataframe
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate cols idx, comp, y_test
    df.to_csv(f"{model_dir}/test_results.csv", index=False)

    preds = np.array(preds)
    y_test = df.y_test.values

    res = preds - y_test
    mae = np.abs(res).mean(axis=1)
    rmse = (res ** 2).mean(axis=1) ** 0.5
    r2 = r2_score(
        y_test[:, None].repeat(ensemble_folds, 1), preds.T, multioutput="raw_values"
    )

    r2_avg = r2.mean()
    r2_std = r2.std()

    mae_avg = mae.mean()
    mae_std = mae.std() / np.sqrt(ensemble_folds)

    rmse_avg = rmse.mean()
    rmse_std = rmse.std() / np.sqrt(ensemble_folds)

    if ensemble_folds == 1:
        print("\nPerformance Metrics:")
        print(f"R2 Score: {r2_avg:.4f} ")
        print(f"MAE: {mae_avg:.4f}")
        print(f"RMSE: {rmse_avg:.4f}")
    else:
        print("\nPerformance Metrics:")
        print(f"R2 Score: {r2_avg:.4f} +/- {r2_std:.4f}")
        print(f"MAE: {mae_avg:.4f} +/- {mae_std:.4f}")
        print(f"RMSE: {rmse_avg:.4f} +/- {rmse_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        y_ens = preds.mean(axis=0)

        mae_ens = np.abs(y_test - y_ens).mean()
        rmse_ens = ((y_test - y_ens) ** 2).mean() ** 0.5

        r2_ens = r2_score(y_test, y_ens)

        print("\nEnsemble Performance Metrics:")
        print(f"R2 Score : {r2_ens:.4f} ")
        print(f"MAE  : {mae_ens:.4f}")
        print(f"RMSE : {rmse_ens:.4f}")

    return r2_avg, mae_avg, rmse_avg
