import gc
from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.functional import softmax
from tqdm import tqdm

from roost.utils import interruptable

from .core import sampled_softmax, save_checkpoint


class BaseModel(nn.Module, ABC):
    """ A base class for regression and classification models. """

    def __init__(
        self, task, n_targets, robust, device=None, epoch=1, best_val_score=None
    ):
        super().__init__()
        self.task = task
        self.robust = robust
        self.device = device
        self.epoch = epoch
        self.best_val_score = best_val_score
        self.val_score_name = "MAE" if task == "regression" else "Acc"
        self.model_params = {}

    @interruptable
    def fit(
        self,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs,
        criterion,
        normalizer,
        model_dir,
        checkpoint=True,
        writer=None,
        verbose=True,
    ):
        start_epoch = self.epoch
        for epoch in range(start_epoch, start_epoch + epochs):
            self.epoch += 1
            # Training
            print(f"Epoch: [{epoch}/{start_epoch + epochs - 1}]")
            train_metrics = self.evaluate(
                train_loader, criterion, optimizer, normalizer, "train", verbose
            )

            if verbose:
                metric_str = "\t ".join(
                    f"{key} {val:.3f}" for key, val in train_metrics.items()
                )
                print(f"Train      : {metric_str}")

            # Validation
            if val_loader is None:
                is_best = False
            else:
                with torch.no_grad():
                    # Evaluate on validation set
                    val_metrics = self.evaluate(
                        val_loader, criterion, None, normalizer, action="val"
                    )

                if verbose:
                    metric_str = "\t ".join(
                        f"{key} {val:.3f}" for key, val in val_metrics.items()
                    )
                    print(f"Validation : {metric_str}")

                if self.task == "regression":
                    val_score = val_metrics["mae"]
                    is_best = val_score < self.best_val_score
                else:  # classification
                    val_score = val_metrics["acc"]
                    is_best = val_score > self.best_val_score

            if is_best:
                self.best_val_score = val_score

            if checkpoint:
                checkpoint_dict = {
                    "task": self.task,
                    "model_params": self.model_params,
                    "state_dict": self.state_dict(),
                    "epoch": self.epoch,
                    "best_val_score": self.best_val_score,
                    "val_score_name": self.val_score_name,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }

                if self.task == "regression":
                    checkpoint_dict["normalizer"] = normalizer.state_dict()

                if hasattr(self, "swa"):
                    checkpoint_dict["swa"] = self.swa.copy()
                    for key in ["model", "scheduler"]:
                        # remove model as it can't and needs not be serialized
                        del checkpoint_dict["swa"][key]
                        state_dict = self.swa[key].state_dict()
                        checkpoint_dict["swa"][f"{key}_state_dict"] = state_dict

                save_checkpoint(checkpoint_dict, is_best, model_dir)

            if writer is not None:
                for metric, val in train_metrics.items():
                    writer.add_scalar(f"training/{metric}", val, epoch + 1)

                if val_loader is not None:
                    for metric, val in val_metrics.items():
                        writer.add_scalar(f"validation/{metric}", val, epoch + 1)

            if hasattr(self, "swa") and epoch > self.swa["start"]:
                self.swa["model"].update_parameters(self)
                self.swa["scheduler"].step()
            else:
                scheduler.step()

            # catch memory leak
            gc.collect()

        # if self.swa:
        #     # handle batch norm + SWA (does nothing if model has no batch norm)
        #     currently incompatible (https://github.com/pytorch/pytorch/issues/49082)
        #     torch.optim.swa_utils.update_bn(train_loader, self.swa["model"])

    def evaluate(
        self, loader, criterion, optimizer, normalizer, action="train", verbose=False
    ):
        """ Evaluate the model for one epoch """

        assert action in ["train", "val"], f"action must be train or val, got {action}"
        self.train() if action == "train" else self.eval()

        # records both regr. and clf. metrics for an epoch to compute averages below
        metrics = {"loss": [], "mae": [], "rmse": [], "acc": [], "f1": []}

        # we do not need batch_comp or batch_ids when training
        for input_, target, *_ in tqdm(loader, disable=not verbose):

            # move tensors to GPU
            input_ = (tensor.to(self.device) for tensor in input_)

            # compute output
            output = self(*input_)

            if self.task == "regression":
                target_norm = normalizer.norm(target)
                target_norm = target_norm.to(self.device)
                if self.robust:
                    mean, log_std = output.chunk(2, dim=1)
                    loss = criterion(mean, log_std, target_norm)

                    pred = normalizer.denorm(mean.data.cpu())
                    std_al = normalizer.std * log_std.exp().data.cpu()
                    ae = (target - pred).abs()
                else:
                    loss = criterion(output, target_norm)
                    pred = normalizer.denorm(output.data.cpu())

                metrics["mae"].append((pred - target).abs().mean())
                metrics["rmse"].append(((pred - target) ** 2).mean() ** 0.5)

            else:  # classification
                target = target.to(self.device).squeeze()
                if self.robust:
                    output, log_std = output.chunk(2, dim=1)
                    logits = sampled_softmax(output, log_std)
                    loss = criterion(torch.log(logits), target)
                else:
                    loss = criterion(output, target)
                    logits = softmax(output, dim=1)
                metrics["acc"].append((logits.argmax(1) == target).float().mean().cpu())

                # call .cpu() for automatic numpy conversion
                # since sklearn metrics need numpy arrays
                f1 = f1_score(logits.argmax(1).cpu(), target.cpu(), average="weighted")
                metrics["f1"].append(f1)

            metrics["loss"].append(loss.cpu().item())

            if action == "train":
                # compute gradient and take an optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = {key: np.array(val).mean() for key, val in metrics.items() if val}

        return metrics

    @torch.no_grad()
    def predict(self, generator, verbose=False, repeat=1):
        """ Generate predictions """

        test_ids = []
        test_comp = []
        test_targets = []
        test_output = []

        # Ensure model is in evaluation mode
        self.eval()
        model = self.swa["model"] if hasattr(self, "swa") else self

        # iterate over mini-batches
        for input_, target, comps, ids in tqdm(generator, disable=not verbose):

            # move tensors to device (GPU or CPU)
            input_ = (t.to(self.device) for t in input_)

            # compute output
            output = model(*input_, repeat=repeat)

            # collect the model outputs
            test_ids += ids
            test_comp += comps
            test_targets.append(target)
            test_output.append(output)

        return (
            test_ids,
            test_comp,
            torch.cat(test_targets, dim=0).view(-1).numpy(),
            torch.cat(test_output, dim=0),
        )

    @torch.no_grad()
    def featurize(self, generator):
        """Generate features for a list of composition strings. When using Roost,
        this runs only the message-passing part of the model without the ResidualNet.

        Args:
            generator (DataLoader): PyTorch loader with the same data format used in fit()

        Returns:
            np.array: 2d array of features
        """
        err_msg = f"{self} needs to be fitted before it can be used for featurization"
        assert self.epoch > 0, err_msg

        self.eval()  # ensure model is in evaluation mode
        features = []

        for input_, *_ in generator:

            input_ = (tensor.to(self.device) for tensor in input_)

            output = self.material_nn(*input_).cpu().numpy()
            features.append(output)

        return np.vstack(features)

    @abstractmethod
    def forward(self, *x):
        """
        Forward pass through the model. Needs to be implemented in any derived
        model class.
        """
        raise NotImplementedError("forward() is not defined!")
