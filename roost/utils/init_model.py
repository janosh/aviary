from os.path import isfile, relpath

import torch
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, NLLLoss
from torch.optim.swa_utils import SWALR, AveragedModel

from roost.core import Normalizer, RobustL1Loss, RobustL2Loss
from roost.segments import ResidualNet
from roost.utils.io import bold


def init_model(
    model_class,
    model_dir,
    model_params,
    loss,
    optim,
    learning_rate,
    weight_decay,
    momentum,
    device=torch.device("cpu"),
    milestones=[],
    gamma=0.3,
    fine_tune=None,
    transfer=None,
    swa=False,
):

    task = model_params["task"]
    robust = model_params["robust"]
    n_targets = model_params["n_targets"]

    model_path = model_dir + "/checkpoint.pth.tar"

    if isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)

        model = model_class(**checkpoint["model_params"], device=device)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])
        if model.task != task:
            print(
                f"Loaded model had task {bold(model.task)}, changing to {bold(task)}.\n",
                bold("Was this intended?"),
            )
            model.task = task

        if fine_tune:
            n_out_err = (
                "cannot fine-tune between tasks with different numbers of outputs"
                " - use transfer option instead"
            )
            assert model.model_params["robust"] == robust, n_out_err
            assert model.model_params["n_targets"] == n_targets, n_out_err
        elif transfer:
            model.model_params["task"] = task
            model.robust = robust
            model.model_params["robust"] = robust
            model.model_params["n_targets"] = n_targets

            # # NOTE currently if you use a model as a feature extractor and then
            # # resume for a checkpoint of that model the material_nn unfreezes.
            # # This is potentially not the behavior a user might expect.
            # for p in model.material_nn.parameters():
            #     p.requires_grad = False

            output_dim = 2 * n_targets if robust else n_targets
            dims = [
                model_params["elem_fea_len"],
                *model_params["out_hidden"],
                output_dim,
            ]

            model.output_nn = ResidualNet(dims, use_mnf=model_params["use_mnf"])
        else:  # default case: resume previous training with no changes to data or model
            # TODO work out how to ensure that we are using the same optimizer
            # when resuming such that the state dictionaries do not clash.

            epoch = model.epoch = checkpoint["epoch"]
            best_score = model.best_val_score = checkpoint["best_val_score"]
            score_name = model.val_score_name = checkpoint["val_score_name"]
            print(
                f"Resuming training from '{bold(relpath(model_path))}' at epoch {epoch}"
            )
            score = f"{score_name} = {best_score:.4g}"
            print(f"Model's previous best validation score: {score}")

    else:  # model_path does not exist, train new model
        print("Training a new model from scratch")
        print(f"Checkpoint will be saved to '{bold(relpath(model_path))}'")
        model = model_class(device=device, **model_params)

        model.to(device)

    # Select optimizer
    assert optim in ["SGD", "Adam", "AdamW"], "Only SGD, Adam or AdamW allowed"
    optim_class = getattr(torch.optim, optim)
    optim_params = {
        "params": model.parameters(),
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    if optim == "SGD":
        optim_params["momentum"] = momentum
    optimizer = optim_class(**optim_params)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )

    # Select Task and Loss Function
    if task == "classification":
        normalizer = None
        criterion = NLLLoss() if robust else CrossEntropyLoss()

    else:  # regression
        normalizer = Normalizer()
        if robust:
            if loss == "L1":
                criterion = RobustL1Loss
            elif loss == "L2":
                criterion = RobustL2Loss
            else:
                raise NameError(
                    "Only L1 or L2 losses are allowed for robust regression tasks"
                )
        else:
            if loss == "L1":
                criterion = L1Loss()
            elif loss == "L2":
                criterion = MSELoss()
            else:
                raise NameError("Only L1 or L2 losses are allowed for regression tasks")

    if isfile(model_path) and not fine_tune and not transfer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if normalizer is not None:
            normalizer.load_state_dict(checkpoint["normalizer"])

        if "swa" in checkpoint.keys():
            model.swa = checkpoint["swa"]

            model_dict = model.swa["model_state_dict"]
            model.swa["model"] = AveragedModel(model)
            model.swa["model"].load_state_dict(model_dict)

            scheduler_dict = model.swa["scheduler_state_dict"]
            model.swa["scheduler"] = SWALR(optimizer, model.swa["lr"])
            model.swa["scheduler"].load_state_dict(scheduler_dict)

    elif swa:  # setup SWA from scratch, i.e. no previous run to load
        swa["model"] = AveragedModel(model)
        swa["scheduler"] = SWALR(optimizer, swa_lr=swa["lr"])
        model.swa = swa

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Number of Trainable Parameters: {num_param:,}")

    # TODO parallelise the code over multiple GPUs. Currently DataParallel
    # crashes as subsets of the batch have different sizes due to the use of
    # lists of lists rather the zero-padding.
    # if (torch.cuda.device_count() > 1) and (device==torch.device("cuda")):
    #     print("The model will use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model.to(device)

    return model, criterion, optimizer, scheduler, normalizer
