import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .utils import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 500,
    lr: float = 2e-3,
    weight_decay: float = 5e-4,
    batch_size: int = 256,
    val_batch_size: int = 256,
    num_workers: int = 2,
    optimizer: str = "adam",                 # "adam" | "sgd"
    momentum: float = 0.9,                   # used for SGD
    scheduler: str = "cosine",               # "none" | "cosine" | "step"
    step_size: int = 50,                     # StepLR
    gamma: float = 0.1,                      # StepLR
    seed: int = 2024,
    **model_kwargs,
):
    # pass only what this model understands
    if model_name == "linear":
        model_kwargs = {}  # linear takes no extra constructor args
    elif model_name in {"mlp", "mlp_deep", "mlp_deep_residual"}:
        allowed = {"hidden_dim", "num_layers", "dropout"}  # example names you’ll support
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed}
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **model_kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_data = load_data("classification_data/val", shuffle=False, batch_size=val_batch_size, num_workers=num_workers)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    if optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # sgd
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )

    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
    elif scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        sched = None

    global_step = 0
    best_val = -1.0
    best_state = None
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # TODO: implement training step
            # forward
            logits = model(img)
            loss = loss_func(logits, label)
            # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            #train metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == label).float().mean()
                metrics["train_acc"].append(acc.item())

            # log training loss every iteration
            logger.add_scalar('train_loss', loss.item(), global_step) 

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        model.eval()
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                # TODO: compute validation accuracy
                logits = model(img)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == label).float().mean()
                metrics["val_acc"].append(acc.item())

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        # log epoch metrics at current global step
        logger.add_scalar('train_accuracy', epoch_train_acc.item(), global_step)
        logger.add_scalar('val_accuracy', epoch_val_acc.item(), global_step)

        # track & save best by val acc
        if epoch_val_acc > best_val:
            best_val = epoch_val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, log_dir / f"{model_name}_best.th")

        if sched is not None:
            sched.step()

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )
    # --- restore best weights and save for grader ---
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val_acc = {best_val:.4f}")

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # logging / IO
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    # core hyperparams
    parser.add_argument("--num_epoch", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
