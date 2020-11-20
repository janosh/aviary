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


def results_classification(
    model_class,
    model_dir,
    ensemble_folds,
    test_set,
    data_params,
    robust,
    device,
    eval_type="checkpoint",
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

    y_pre_logits = []
    y_logits = []
    if robust:
        y_pre_ale = []

    acc, roc_auc, precision, recall, fscore = np.zeros([5, ensemble_folds])

    for ens in range(ensemble_folds):

        if ensemble_folds == 1:
            checkpoint_path = f"{model_dir}/{eval_type}.pth.tar"
        else:
            checkpoint_path = f"{model_dir}/ens_{ens}/{eval_type}.pth.tar"
            print(f"Evaluating Model {ens + 1}/{ensemble_folds}")

        assert isfile(checkpoint_path), f"no checkpoint found at '{checkpoint_path}'"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        assert (
            checkpoint["model_params"]["robust"] == robust
        ), f"robustness of checkpoint '{checkpoint_path}' is not {robust}"

        model = model_class(**checkpoint["model_params"], device=device)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        with torch.no_grad():
            idx, comp, y_test, output = model.predict(test_generator)

        if model.robust:
            mean, log_std = output.chunk(2, dim=1)
            logits = sampled_softmax(mean, log_std, samples=10).data.cpu().numpy()
            pre_logits = mean.data.cpu().numpy()
            pre_logits_std = torch.exp(log_std).data.cpu().numpy()
            y_pre_ale.append(pre_logits_std)
        else:
            pre_logits = output.data.cpu().numpy()

        logits = softmax(pre_logits, axis=1)

        y_pre_logits.append(pre_logits)
        y_logits.append(logits)

        y_test_ohe = np.zeros_like(pre_logits)
        y_test_ohe[np.arange(y_test.size), y_test] = 1

        acc[ens] = accuracy_score(y_test, np.argmax(logits, axis=1))
        roc_auc[ens] = roc_auc_score(y_test_ohe, logits)
        precision[ens], recall[ens], fscore[ens] = precision_recall_fscore_support(
            y_test, np.argmax(logits, axis=1), average="weighted"
        )[:3]

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
        print("\nModel Performance Metrics:")
        print(f"Accuracy : {acc_avg:.4f}")
        print(f"ROC-AUC  : {roc_auc_avg:.4f}")
        print(f"Weighted Precision : {precision_avg:.4f}")
        print(f"Weighted Recall    : {recall_avg:.4f}")
        print(f"Weighted F-score   : {fscore_avg:.4f}")
    else:

        print("\nModel Performance Metrics:")
        print(f"Accuracy : {acc_avg:.4f} +/- {acc_std:.4f}")
        print(f"ROC-AUC  : {roc_auc_avg:.4f} +/- {roc_auc_std:.4f}")
        print(f"Weighted Precision : {precision_avg:.4f} +/- {precision_std:.4f}")
        print(f"Weighted Recall    : {recall_avg:.4f} +/- {recall_std:.4f}")
        print(f"Weighted F-score   : {fscore_avg:.4f} +/- {fscore_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        ens_logits = np.mean(y_logits, axis=0)

        y_test_ohe = np.zeros_like(ens_logits)
        y_test_ohe[np.arange(y_test.size), y_test] = 1

        ens_acc = accuracy_score(y_test, np.argmax(ens_logits, axis=1))
        ens_roc_auc = roc_auc_score(y_test_ohe, ens_logits)
        ens_precision, ens_recall, ens_fscore = precision_recall_fscore_support(
            y_test, np.argmax(ens_logits, axis=1), average="weighted"
        )[:3]

        print("\nEnsemble Performance Metrics:")
        print(f"Accuracy : {ens_acc:.4f} ")
        print(f"ROC-AUC  : {ens_roc_auc:.4f}")
        print(f"Weighted Precision : {ens_precision:.4f}")
        print(f"Weighted Recall    : {ens_recall:.4f}")
        print(f"Weighted F-score   : {ens_fscore:.4f}")

    # NOTE we save pre_logits rather than logits due to fact that with the
    # heteroscedastic setup we want to be able to sample from the gaussian
    # distributed pre_logits we parameterise.
    core = {"id": idx, "composition": comp, "target": y_test}

    results = {}
    for n_ens, y_pre_logit in enumerate(y_pre_logits):
        pred_dict = {
            f"class-{lab}-pred_{n_ens}": val for lab, val in enumerate(y_pre_logit.T)
        }
        results.update(pred_dict)
        if robust:
            ale_dict = {
                f"class-{lab}-ale_{n_ens}": val
                for lab, val in enumerate(y_pre_ale[n_ens].T)
            }
            results.update(ale_dict)

    df = pd.DataFrame({**core, **results})

    if ensemble_folds == 1:
        df.to_csv(f"{model_dir}/test_results.csv", index=False)
    else:
        df.to_csv(f"{model_dir}/ensemble_results.csv", index=False)

    return acc_avg, roc_auc_avg, precision_avg, recall_avg, fscore_avg
