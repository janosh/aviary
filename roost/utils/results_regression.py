from os.path import isfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from roost import plots
from roost.core import ROOT, Normalizer


def results_regression(
    model_class,
    model_name,
    run_id,
    ensemble_folds,
    test_set,
    data_params,
    robust,
    device=torch.device("cpu"),
    eval_type="checkpoint",
    repeat=1,  # use with MNF to get epistemic uncertainty
):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    print(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        "------------Evaluate model on Test Set------------\n"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    )

    test_generator = DataLoader(test_set, **data_params)
    # y_ale only needed if robust is True
    y_ensemble, y_ale = np.zeros([2, ensemble_folds, len(test_set)])

    save_dir = f"{ROOT}/models/{model_name}"

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            checkpoint_path = f"{save_dir}/{eval_type}-r{run_id}.pth.tar"
        else:
            checkpoint_path = f"{save_dir}/{eval_type}-r{j}.pth.tar"
            print(f"Evaluating Model {j + 1}/{ensemble_folds}")

        assert isfile(checkpoint_path), f"no checkpoint found at '{checkpoint_path}'"
        checkpoint = torch.load(checkpoint_path, map_location=device)

        assert (
            checkpoint["model_params"]["robust"] == robust
        ), f"robustness of checkpoint '{checkpoint_path}' is not {robust}"

        model = model_class(**checkpoint["model_params"], device=device)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        normalizer = Normalizer()
        normalizer.load_state_dict(checkpoint["normalizer"])

        with torch.no_grad():
            idx, comp, y_test, output = model.predict(test_generator, repeat=repeat)

        output = (
            output.data.cpu().squeeze()
        )  # move preds to CPU in case model ran on GPU
        if robust:
            mean, log_std_al = [x.squeeze() for x in output.chunk(2, dim=1)]
            if repeat > 1:
                log_std_al = log_std_al.mean(-1)
                std_ep = (mean.std(-1) * normalizer.std).numpy()
                mean = mean.mean(-1)

            pred = normalizer.denorm(mean).numpy()
            std_al = (log_std_al.exp() * normalizer.std).numpy()
            y_ale[j, :] = std_al
        else:
            if repeat > 1:
                std_ep = (output.std(-1) * normalizer.std).numpy() * 100
                output = output.mean(-1)
            pred = normalizer.denorm(output).numpy()

        y_ensemble[j, :] = pred

    res = y_ensemble - y_test
    mae = np.abs(res).mean(axis=1)
    rmse = (res ** 2).mean(axis=1) ** 0.5
    r2 = r2_score(
        np.repeat(y_test[:, None], ensemble_folds, axis=1),
        y_ensemble.T,
        multioutput="raw_values",
    )

    r2_avg = r2.mean()
    r2_std = r2.std()

    mae_avg = mae.mean()
    mae_std = mae.std() / np.sqrt(mae.shape[0])

    rmse_avg = rmse.mean()
    rmse_std = rmse.std() / np.sqrt(rmse.shape[0])

    if ensemble_folds == 1:
        print("\nModel Performance Metrics:")
        print(f"R2 Score: {r2_avg:.4f} ")
        print(f"MAE: {mae_avg:.4f}")
        print(f"RMSE: {rmse_avg:.4f}")
    else:
        print("\nModel Performance Metrics:")
        print(f"R2 Score: {r2_avg:.4f} +/- {r2_std:.4f}")
        print(f"MAE: {mae_avg:.4f} +/- {mae_std:.4f}")
        print(f"RMSE: {rmse_avg:.4f} +/- {rmse_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        y_ens = y_ensemble.mean(axis=0)

        mae_ens = np.abs(y_test - y_ens).mean()
        rmse_ens = ((y_test - y_ens) ** 2).mean() ** 0.5

        r2_ens = r2_score(y_test, y_ens)

        print("\nEnsemble Performance Metrics:")
        print(f"R2 Score : {r2_ens:.4f} ")
        print(f"MAE  : {mae_ens:.4f}")
        print(f"RMSE : {rmse_ens:.4f}")

    core = {"id": idx, "composition": comp, "target": y_test}
    results = {f"pred_{n}": val for (n, val) in enumerate(y_ensemble)}
    if robust:
        ale = {f"std_al_{n}": val for (n, val) in enumerate(y_ale)}
        results.update(ale)
    if repeat > 1:
        results.update({f"std_ep_repeat_{repeat}": std_ep})
    if robust and repeat > 1:
        results.update({"std_tot": (std_ep ** 2 + std_al ** 2) ** 0.5})

    df = pd.DataFrame({**core, **results})

    if ensemble_folds == 1:
        df.to_csv(f"{save_dir}/test_results-r{run_id}.csv", index=False)
    else:
        df.to_csv(f"{save_dir}/ensemble_results-r{run_id}.csv", index=False)

    return r2_avg, mae_avg, rmse_avg
