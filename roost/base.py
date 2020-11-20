import gc
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax
from tqdm.autonotebook import trange

from roost.utils import interruptable

from .core import (
    AverageMeter,
    ClassificationMetrics,
    RegressionMetrics,
    sampled_softmax,
    save_checkpoint,
)


class BaseModel(nn.Module, ABC):
    """
    A base class for models.
    """

    def __init__(
        self, task, n_targets, robust, device=None, epoch=1, best_val_score=None
    ):
        super().__init__()
        self.task = task
        self.robust = robust
        self.device = device
        self.epoch = epoch
        self.best_val_score = best_val_score
        self.model_params = {}

    @interruptable
    def fit(
        self,
        train_generator,
        val_generator,
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
            t_loss, t_metrics = self.evaluate(
                generator=train_generator,
                criterion=criterion,
                optimizer=optimizer,
                normalizer=normalizer,
                action="train",
                verbose=verbose,
            )

            if verbose:
                print(f"Epoch: [{epoch}/{start_epoch + epochs - 1}]")
                metric_str = "\t".join(
                    f"{key} {val:.3f}" for key, val in t_metrics.items()
                )
                print(f"Train      : Loss {t_loss:.4f}\t{metric_str}")

            # Validation
            if val_generator is None:
                is_best = False
            else:
                with torch.no_grad():
                    # evaluate on validation set
                    v_loss, v_metrics = self.evaluate(
                        generator=val_generator,
                        criterion=criterion,
                        optimizer=None,
                        normalizer=normalizer,
                        action="val",
                    )

                if verbose:
                    metric_str = "\t".join(
                        f"{key} {val:.3f}" for key, val in v_metrics.items()
                    )
                    print(f"Validation : Loss {v_loss:.4f}\t{metric_str}")

                if self.task == "regression":
                    val_score = v_metrics["MAE"]
                    is_best = val_score < self.best_val_score
                elif self.task == "classification":
                    val_score = v_metrics["Acc"]
                    is_best = val_score > self.best_val_score

            if is_best:
                self.best_val_score = val_score

            if checkpoint:
                checkpoint_dict = {
                    "model_params": self.model_params,
                    "state_dict": self.state_dict(),
                    "epoch": self.epoch,
                    "best_val_score": self.best_val_score,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if self.task == "regression":
                    checkpoint_dict.update({"normalizer": normalizer.state_dict()})

                save_checkpoint(checkpoint_dict, is_best, model_dir)

            if writer is not None:
                writer.add_scalar("train/loss", t_loss, epoch + 1)
                for metric, val in t_metrics.items():
                    writer.add_scalar(f"train/{metric}", val, epoch + 1)

                if val_generator is not None:
                    writer.add_scalar("validation/loss", v_loss, epoch + 1)
                    for metric, val in v_metrics.items():
                        writer.add_scalar(f"validation/{metric}", val, epoch + 1)

            scheduler.step()

            # catch memory leak
            gc.collect()

    def evaluate(
        self, generator, criterion, optimizer, normalizer, action="train", verbose=False
    ):
        """evaluate the model"""

        if action == "val":
            self.eval()
        elif action == "train":
            self.train()
        else:
            raise NameError("Only train or val allowed as action")

        loss_meter = AverageMeter()

        if self.task == "regression":
            metric_meter = RegressionMetrics()
        elif self.task == "classification":
            metric_meter = ClassificationMetrics()
        else:
            raise ValueError(f"invalid task: {self.task}")

        with trange(len(generator), disable=(not verbose)) as t:
            # we do not need batch_comp or batch_ids when training
            for input_, target, _, _ in generator:

                # move tensors to GPU
                input_ = (tensor.to(self.device) for tensor in input_)

                if self.task == "regression":
                    # normalize target if needed
                    target_norm = normalizer.norm(target)
                    target_norm = target_norm.to(self.device)
                else:  # classification
                    target = target.to(self.device)

                # compute output
                output = self(*input_)

                if self.task == "regression":
                    if self.robust:
                        output, log_std = output.chunk(2, dim=1)
                        loss = criterion(output, log_std, target_norm)
                    else:
                        loss = criterion(output, target_norm)

                    pred = normalizer.denorm(output.data.cpu())
                    metric_meter.update(
                        pred.data.cpu().numpy(), target.data.cpu().numpy()
                    )

                elif self.task == "classification":
                    if self.robust:
                        output, log_std = output.chunk(2, dim=1)
                        logits = sampled_softmax(output, log_std)
                        loss = criterion(torch.log(logits), target.squeeze(1))
                    else:
                        loss = criterion(output, target.squeeze(1))
                        logits = softmax(output, dim=1)

                    # classification metrics from sklearn need numpy arrays
                    metric_meter.update(
                        logits.data.cpu().numpy(), target.data.cpu().numpy()
                    )

                loss_meter.update(loss.data.cpu().item())

                if action == "train":
                    # compute gradient and take an optimizer step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                t.update()

        return loss_meter.avg, metric_meter.metric_dict

    def predict(self, generator, verbose=False, repeat=1):
        """
        evaluate the model
        """

        test_ids = []
        test_comp = []
        test_targets = []
        test_output = []

        # Ensure model is in evaluation mode
        self.eval()

        with torch.no_grad():
            with trange(len(generator), disable=(not verbose)) as pbar:
                for input_, target, batch_comp, batch_ids in generator:

                    # move tensors to device (GPU or CPU)
                    input_ = (t.to(self.device) for t in input_)

                    # compute output
                    output = self(*input_, repeat=repeat)

                    # collect the model outputs
                    test_ids += batch_ids
                    test_comp += batch_comp
                    test_targets.append(target)
                    test_output.append(output)

                    pbar.update()

        return (
            test_ids,
            test_comp,
            torch.cat(test_targets, dim=0).view(-1).numpy(),
            torch.cat(test_output, dim=0),
        )

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

        with torch.no_grad():
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
