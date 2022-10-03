"""Main script to run the training of the model(ECGNet, Resnet101).
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


from typing import Tuple
import os
import random
import argparse
import json
from tqdm import tqdm
import numpy as np
import wandb
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

from preprocessing.preprocess import preprocess
from utils.torch_dataloader import DataGen
from utils.metrics import Metrics

# Random seed
seed = 42
random.seed(seed)
np.random.seed(seed)


def dump_logs(train_results: tuple, test_results: tuple, name: str):
    """Dumps the performance logs to a json file.

    Parameters
    ----------
    train_results: tuple
        Training results.
    test_results: tuple
        Testing results.
    name: str
        Name of the model.

    """

    logs = {
        "train_loss": train_results[0],
        "train_mean_accuracy": train_results[1],
        "train_roc_score": train_results[2],
        "test_loss": test_results[0],
        "test_mean_accuracy": test_results[1],
        "test_roc_score": test_results[2],
    }
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)

    with open(os.path.join(logs_path, f"{name}_train_logs.json"), "w") as json_file:
        json.dump(logs, json_file)


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim,
    loss_func,
    dataset,
    epoch: int,
    device: torch.device,
    loggr: bool = False,
) -> Tuple[float, float, float]:
    """Training of the model for one epoch.

    Parameters
    ----------
    model: nn.Module
        Model to be trained.
    optimizer: torch.optim
        Optimizer to be used.
    loss_func: torch.nn._Loss
        Loss function to be used.
    dataset: torch.utils.data.DataLoader
        Dataset to be used.
    epoch: int, optional
        The current epoch.
    device: torch.device
        Device to be used.
    loggr: bool, optional
        To log wandb metrics. (default: False)

    """

    model.train()

    pred_all = []
    loss_all = []
    gt_all = []

    for batch_step in tqdm(range(len(dataset)), desc="train"):
        batch_x, batch_y = dataset[batch_step]
        batch_x = batch_x.to(device)
        batch_x = batch_x.permute(0, 2, 1)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        pred_all.append(pred.cpu().detach().numpy())
        loss = loss_func(pred, batch_y)
        loss_all.append(loss.cpu().detach().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        gt_all.extend(batch_y.cpu().detach().numpy())

    print("Epoch: {0}".format(epoch))
    print("Train loss: ", np.mean(loss_all), end="\n" * 2)

    pred_all = np.concatenate(pred_all, axis=0)
    _, mean_acc = Metrics(np.array(gt_all), pred_all)
    roc_score = roc_auc_score(np.array(gt_all), pred_all, average="macro")

    if loggr is not None:
        loggr.log({"train_mean_accuracy": mean_acc, "epoch": epoch})
        loggr.log({"train_roc_score": roc_score, "epoch": epoch})
        loggr.log({"train_loss": np.mean(loss_all), "epoch": epoch})

    return np.mean(loss_all), mean_acc, roc_score


def test_epoch(
    model: nn.Module,
    loss_func: torch.optim,
    dataset,
    epoch: int,
    device: torch.device,
    loggr: bool = False,
) -> Tuple[float, float, float]:
    """Testing of the model for one epoch.

    Parameters
    ----------
    model: nn.Module
        Model to be trained.
    loss_func: torch.nn.BCEWithLogitsLoss
        Loss function to be used.
    dataset: torch.utils.data.DataLoader
        Dataset to be used.
    epoch: int, optional
        The current epoch.
    device: torch.device
        Device to be used.
    loggr: bool, optional
        To log wandb metrics. (default: False)

    """

    model.eval()

    pred_all = []
    loss_all = []
    gt_all = []

    for batch_step in tqdm(range(len(dataset)), desc="valid"):
        batch_x, batch_y = dataset[batch_step]
        batch_x = batch_x.to(device)
        batch_x = batch_x.permute(0, 2, 1)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        pred_all.append(pred.cpu().detach().numpy())
        loss = loss_func(pred, batch_y)
        loss_all.append(loss.cpu().detach().numpy())
        gt_all.extend(batch_y.cpu().detach().numpy())

    print("Test loss: ", np.mean(loss_all))
    pred_all = np.concatenate(pred_all, axis=0)
    _, mean_acc = Metrics(np.array(gt_all), pred_all)
    roc_score = roc_auc_score(np.array(gt_all), pred_all, average="macro")

    if loggr is not None:
        loggr.log({"test_mean_accuracy": mean_acc, "epoch": epoch})
        loggr.log({"test_roc_score": roc_score, "epoch": epoch})
        loggr.log({"test_loss": np.mean(loss_all), "epoch": epoch})

    return np.mean(loss_all), mean_acc, roc_score


def train(
    model: nn.Module,
    path: str = "data/ptb",
    batch_size: int = 32,
    epochs: int = 60,
    loggr: wandb = None,
    name: str = "ecgnet",
) -> None:
    """Data preprocessing and training of the model.

    Parameters
    ----------
    model: nn.Module
        Model to be trained.
    path: str, optional
        Path to the directory containing the data. (default: 'data/ptb')
    batch_size: int, optional
        Batch size. (default: 32)
    epochs: int, optional
        Number of epochs. (default: 60)
    loggr: wandb, optional
        To log wandb metrics. (default: None)
    name: str, optional
        Name of the model. (default: 'ecgnet')

    """

    X_train_scale, y_train, _, _, X_val_scale, y_val = preprocess(path=path)
    train_gen = DataGen(X_train_scale, y_train, batch_size=batch_size)
    val_gen = DataGen(X_val_scale, y_val, batch_size=batch_size)

    checkpoint_filepath = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_filepath, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Distributed Training if for multiple GPUs 
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() == 1 :
        print("You have a GPU on your system, Let's use it")
    else:
        print("You don't have any GPU available on your system, Let's use CPU")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.BCEWithLogitsLoss()

    best_score = 0.0
    for epoch in range(epochs):
        train_results = train_epoch(
            model, optimizer, loss_func, train_gen, epoch, device, loggr=loggr
        )
        test_results = test_epoch(model, loss_func, val_gen, epoch, device, loggr=loggr)

        if epoch > 5 and best_score < test_results[2]:
            save_path = os.path.join(checkpoint_filepath, f"{name}_weights.pt")
            torch.save(model.state_dict(), save_path)
            dump_logs(train_results, test_results, name)


if __name__ == "__main__":
    """Main function to run the training of the model."""

    # Args parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/ptb", help="Ptb-xl dataset location"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ecgnet",
        help="Select the model to train. (ecgnet, resnet101)",
    )
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument(
        "--loggr", type=bool, default=False, help="Enable wandb logging"
    )
    args = parser.parse_args()

    if args.model == "ecgnet":
        from models.ECGNet import ECGNet

        model = ECGNet()
    else:
        from models.resnet101 import resnet101

        model = resnet101()

    if args.loggr:
        import wandb

        wandb = wandb.init(
            project="IMLE-Net",
            name=args.model,
            notes=f"Model: {args.model} with batch size: {args.batchsize} and epochs: {args.epochs}",
            save_code=True,
        )
        logger = wandb
    else:
        logger = None

    train(
        model,
        path=args.data_dir,
        batch_size=args.batchsize,
        epochs=args.epochs,
        loggr=logger,
        name=args.model,
    )
