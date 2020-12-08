from os.path import isfile

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from roost.core import sampled_softmax


def predict(model_class, test_set, checkpoint_path, device, robust):

    assert isfile(checkpoint_path), f"no checkpoint found at '{checkpoint_path}'"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    chk_robust = checkpoint["model_params"]["robust"]
    assert (
        chk_robust == robust
    ), f"checkpoint['robust'] != robust ({chk_robust} vs {robust})"

    model = model_class(**checkpoint["model_params"], device=device)
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"])

    idx, comp, y_test, output = model.predict(test_set)

    df = pd.DataFrame({"idx": idx, "comp": comp, "y_test": y_test})

    if model.robust:
        mean, log_std = output.chunk(2, dim=1)
        pre_logits_std = torch.exp(log_std).cpu().numpy()
        logits = sampled_softmax(mean, log_std, samples=10).cpu().numpy()
        pre_logits = mean.cpu().numpy()
        for idx, std_al in enumerate(pre_logits_std.T):
            df[f"class_{idx}_std_al"] = std_al

    else:
        pre_logits = output.cpu().numpy()
        logits = softmax(pre_logits, axis=1)

    for idx, (logit, pre_logit) in enumerate(zip(logits.T, pre_logits.T)):
        df[f"class_{idx}_logit"] = logit
        df[f"class_{idx}_pred"] = pre_logit

    return df, y_test, logits, pre_logits


def classification_test(
    model_class,
    model_dir,
    ensemble_folds,
    test_set,
    data_params,
    robust,
    device=torch.device("cpu"),
    eval_type="checkpoint",
):
    """Evaluate an ensemble's performance on the test set"""

    print("\n------------ Evaluating model on test set ------------\n")

    test_set = DataLoader(test_set, **data_params)

    dfs = []
    acc, roc_auc, precision, recall, fscore = np.zeros([5, ensemble_folds])

    for ens in range(ensemble_folds):
        if ensemble_folds > 1:
            print(f"Evaluating Model {ens + 1}/{ensemble_folds}")
            checkpoint_path = f"{model_dir}/ens_{ens}/{eval_type}.pth.tar"
        else:
            checkpoint_path = f"{model_dir}/{eval_type}.pth.tar"

        df_i, y_test, logits, pre_logits = predict(
            model_class, test_set, checkpoint_path, device, robust
        )

        out_cols = [col for col in df_i.columns if col not in ["idx", "comp", "y_test"]]
        col_map = {key: f"{key}_ens_{idx}" for idx, key in enumerate(out_cols)}
        dfs.append(df_i.rename(columns=col_map))

        y_test_ohe = np.zeros_like(pre_logits)
        y_test_ohe[np.arange(len(y_test)), y_test] = 1

        acc[ens] = accuracy_score(y_test, np.argmax(logits, axis=1))
        roc_auc[ens] = roc_auc_score(y_test_ohe, logits)
        precision[ens], recall[ens], fscore[ens] = precision_recall_fscore_support(
            y_test, np.argmax(logits, axis=1), average="weighted"
        )[:3]

    df = pd.concat(dfs, axis=1)  # combine all model outputs into single dataframe
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate cols idx, comp, y_test
    df.to_csv(f"{model_dir}/test_results.csv", index=False)

    acc_avg = acc.mean()
    acc_std = acc.std() / np.sqrt(acc.shape[0])

    roc_auc_avg = roc_auc.mean()
    roc_auc_std = roc_auc.std() / np.sqrt(roc_auc.shape[0])

    precision_avg = precision.mean()
    precision_std = precision.std() / np.sqrt(precision.shape[0])

    recall_avg = recall.mean()
    recall_std = recall.std() / np.sqrt(recall.shape[0])

    fscore_avg = fscore.mean()
    fscore_std = fscore.std() / np.sqrt(fscore.shape[0])

    if ensemble_folds == 1:
        print("\nPerformance Metrics:")
        print(f"Accuracy : {acc_avg:.4f}")
        print(f"ROC-AUC  : {roc_auc_avg:.4f}")
        print(f"Weighted Precision : {precision_avg:.4f}")
        print(f"Weighted Recall    : {recall_avg:.4f}")
        print(f"Weighted F-score   : {fscore_avg:.4f}")
    else:

        print("\nPerformance Metrics:")
        print(f"Accuracy : {acc_avg:.4f} +/- {acc_std:.4f}")
        print(f"ROC-AUC  : {roc_auc_avg:.4f} +/- {roc_auc_std:.4f}")
        print(f"Weighted Precision : {precision_avg:.4f} +/- {precision_std:.4f}")
        print(f"Weighted Recall    : {recall_avg:.4f} +/- {recall_std:.4f}")
        print(f"Weighted F-score   : {fscore_avg:.4f} +/- {fscore_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        ens_logits = np.mean(df.y_logits, axis=0)

        y_test_ohe = np.zeros_like(ens_logits)
        y_test_ohe[np.arange(len(df.y_test)), df.y_test] = 1

        ens_acc = accuracy_score(df.y_test, np.argmax(ens_logits, axis=1))
        ens_roc_auc = roc_auc_score(y_test_ohe, ens_logits)
        ens_precision, ens_recall, ens_fscore = precision_recall_fscore_support(
            df.y_test, np.argmax(ens_logits, axis=1), average="weighted"
        )[:3]

        print("\nEnsemble Performance Metrics:")
        print(f"Accuracy : {ens_acc:.4f} ")
        print(f"ROC-AUC  : {ens_roc_auc:.4f}")
        print(f"Weighted Precision : {ens_precision:.4f}")
        print(f"Weighted Recall    : {ens_recall:.4f}")
        print(f"Weighted F-score   : {ens_fscore:.4f}")

    return acc_avg, roc_auc_avg, precision_avg, recall_avg, fscore_avg
