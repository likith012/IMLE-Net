"""Main script to run the training of the model(imle_net, mousavi, rajpurkar).
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import logging, os, random, argparse
from utils.tf_utils import set_tf_loglevel, str2bool

set_tf_loglevel(logging.ERROR)

import json
import numpy as np
import tensorflow as tf

from preprocessing.preprocess import preprocess
from utils.dataloader import DataGen
from utils.callbacks import model_checkpoint

# Random seed
seed = 42
random.seed(seed)
np.random.seed(seed)


def train(
    model,
    path: str = "data/ptb",
    batch_size: int = 32,
    epochs: int = 60,
    loggr = None,
    name: str = "imle_net",
) -> None:
    """Data preprocessing and training of the model.

    Parameters
    ----------
    model: tf.keras.Model
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
        Name of the model. (default: 'imle_net')

    """

    X_train_scale, y_train, _, _, X_val_scale, y_val = preprocess(path=path)
    train_gen = DataGen(X_train_scale, y_train, batch_size=batch_size)
    val_gen = DataGen(X_val_scale, y_val, batch_size=batch_size)

    metric = "auc"
    checkpoint_filepath = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_filepath, exist_ok=True)

    checkpoint = model_checkpoint(
        checkpoint_filepath, val_gen, loggr=loggr, monitor=metric, name=name
    )
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor=metric,
        min_delta=0.001,
        patience=10,
        mode="auto",
        restore_best_weights=True,
    )

    callbacks = [checkpoint, stop_early]
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        workers=5,
    )
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)

    with open(os.path.join(logs_path, f"{name}_train_logs.json"), 'w') as json_file:
        json.dump(history.history, json_file)


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
        default="imle_net",
        help="Select the model to train. (imle_net, mousavi, rajpurkar)",
    )
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument(
        "--loggr", type=str2bool, default=False, help="Enable wandb logging"
    )
    args = parser.parse_args()

    if args.model == "imle_net":
        from models.IMLENet import build_imle_net
        from configs.imle_config import Config

        model = build_imle_net(Config())
    elif args.model == "mousavi":
        from models.mousavi import build_mousavi
        from configs.mousavi_config import Config

        model = build_mousavi(Config())
    elif args.model == "rajpurkar":
        from models.rajpurkar import build_rajpurkar
        from configs.rajpurkar_config import params

        model = build_rajpurkar(**params)

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
