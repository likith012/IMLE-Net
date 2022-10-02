"""Main script to run the testing of the model(ECGNet, Resnet101).
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


from typing import List
import os
import random
import argparse
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

from preprocessing.preprocess import preprocess
from utils.torch_dataloader import DataGen
from utils.metrics import Metrics, AUC, metric_summary

# Random seed
seed = 42
random.seed(seed)
np.random.seed(seed)


def epoch_run(
    model: nn.Module, dataset: torch.utils.data.Dataset, device: torch.device
) -> List[np.array]:
    """Testing of the model.

    Parameters
    ----------
    model: nn.Module
        Model to be tested.
    dataset: torch.utils.data.DataLoader
        Dataset to be tested.
    device: torch.device
        Device to be used.

    Returns
    -------
    np.array
        Predicted values.

    """

    model.to(device)
    model.eval()
    pred_all = []

    for batch_step in tqdm(range(len(dataset)), desc="test"):
        batch_x, _ = dataset[batch_step]
        batch_x = batch_x.permute(0, 2, 1).to(device)
        pred = model(batch_x)
        pred_all.append(pred.detach().cpu().numpy())
    pred_all = np.concatenate(pred_all, axis=0)

    return pred_all


def test(
    model: nn.Module,
    path: str = "data/ptb",
    batch_size: int = 32,
    name: str = "imle_net",
) -> None:
    """Data preprocessing and testing of the model.

    Parameters
    ----------
    model: nn.Module
        Model to be trained.
    path: str, optional
        Path to the directory containing the data. (default: 'data/ptb')
    batch_size: int, optional
        Batch size. (default: 32)
    name: str, optional
        Name of the model. (default: 'imle_net')

    """

    _, _, X_test_scale, y_test, _, _ = preprocess(path=path)
    test_gen = DataGen(X_test_scale, y_test, batch_size=batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred = epoch_run(model, test_gen, device)

    roc_score = roc_auc_score(y_test, pred, average="macro")
    acc, mean_acc = Metrics(y_test, pred)
    class_auc = AUC(y_test, pred)
    summary = metric_summary(y_test, pred)

    print(f"class wise accuracy: {acc}")
    print(f"accuracy: {mean_acc}")
    print(f"roc_score : {roc_score}")
    print(f"class wise AUC : {class_auc}")
    print(f"F1 score (Max): {summary[0]}")
    print(f"class wise precision, recall, f1 score : {summary}")

    logs = dict()
    logs["roc_score"] = roc_score
    logs["mean_acc"] = mean_acc
    logs["accuracy"] = acc
    logs["class_auc"] = class_auc
    logs["F1 score (Max)"] = summary[0]
    logs["class_precision_recall_f1"] = summary
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)

    with open(os.path.join(logs_path, f"{name}_test_logs.json"), "w") as json_file:
        json.dump(logs, json_file)


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
    args = parser.parse_args()

    if args.model == "ecgnet":
        from models.ECGNet import ECGNet

        model = ECGNet()
    elif args.model == "resnet101":
        from models.resnet101 import resnet101

        model = resnet101()

    path_weights = os.path.join(os.getcwd(), "checkpoints", f"{args.model}_weights.pt")
    model.load_state_dict(torch.load(path_weights))

    test(model, path=args.data_dir, batch_size=args.batchsize, name=args.model)
