from os.path import isfile

import torch
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, NLLLoss

from roost.core import ROOT
from roost.core import Normalizer, RobustL1Loss, RobustL2Loss
from roost.segments import ResidualNetwork


def init_model(
    model_class,
    model_name,
    model_params,
    run_id,
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
):

    task = model_params["task"]
    robust = model_params["robust"]
    n_targets = model_params["n_targets"]
    model_path = f"{ROOT}/models/{model_name}/checkpoint-r{run_id}.pth.tar"

    if isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model = model_class(**checkpoint["model_params"], device=device)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        if fine_tune:
            n_out_err = (
                "cannot fine-tune between tasks with different numbers of outputs"
                " - use transfer option instead"
            )
            assert model.model_params["robust"] == robust, n_out_err
            assert model.model_params["n_targets"] == n_targets, n_out_err
        elif transfer:
            model.task = task
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

            model.output_nn = ResidualNetwork(dims, use_mnf=model_params["use_mnf"])
        else:  # default case: resume previous training with no changes to data or model
            # TODO work out how to ensure that we are using the same optimizer
            # when resuming such that the state dictionaries do not clash.

            print(f"Resuming training from '{model_path}'")
            model.epoch = checkpoint["epoch"]
            model.best_val_score = checkpoint["best_val_score"]

    else:
        print("Training a new model scratch")
        print(f"Checkpoints will be saved to {model_path}")
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

    elif task == "regression":
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
        normalizer.load_state_dict(checkpoint["normalizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

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
