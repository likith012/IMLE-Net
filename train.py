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
from preprocessing.sub_preprocess import preprocess_sub_disease
from utils.dataloader import DataGen
from utils.callbacks import model_checkpoint

# Random seed
seed = 42
random.seed(seed)
np.random.seed(seed)

def train(
    model,
    path: str = "data/ptb",
    batch_size: int = 32, # Reduce this to 16 if there is any memory problem
    epochs: int = 60, 
    loggr=None,
    name: str = "imle_net",
    sub_disease: bool = False,
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
    sub_disease: bool, optional
        If true, the model is trained with subdisease of MI with pretrained weights from main dataset. (default: False)

    """

    metric = "val_auc"
    checkpoint_filepath = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_filepath, exist_ok=True)

    if sub_disease:
        X_train, y_train, X_test, y_test = preprocess_sub_disease(path="data/ptb")
        train_gen = DataGen(X_train, y_train, batch_size=batch_size)
        test_gen = DataGen(X_test, y_test, batch_size=batch_size)
        checkpoint = model_checkpoint(
            checkpoint_filepath,
            test_gen,
            loggr=loggr,
            monitor=metric,
            name=name,
            sub=sub_disease,
        )
    else:
        X_train_scale, y_train, _, _, X_val_scale, y_val = preprocess(path=path)
        train_gen = DataGen(X_train_scale, y_train, batch_size=batch_size)
        val_gen = DataGen(X_val_scale, y_val, batch_size=batch_size)
        checkpoint = model_checkpoint(
            checkpoint_filepath,
            val_gen,
            loggr=loggr,
            monitor=metric,
            name=name,
            sub=sub_disease,
        )

    # Early Stopping
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor=metric,
        min_delta=0.0001,
        patience=20,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    callbacks = [checkpoint, stop_early]

    if sub_disease:
        try:
            path_weights = os.path.join(
                os.getcwd(), "checkpoints", f"{name}_weights.h5"
            )
            model.load_weights(path_weights)
        except:
            raise Exception(
                "Model weights file not found, please train the model on main dataset first."
            )

        if name == "imle_net":
            outputs = tf.keras.layers.Dense(3, activation="softmax")(
                model.layers[-2].output[0]
            )
        else:
            outputs = tf.keras.layers.Dense(3, activation="softmax")(
                model.layers[-2].output
            )

        model = tf.keras.models.Model(inputs=model.input, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        model._name = f"{name}-sub-diagnostic"
        print(model.summary())

        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=test_gen,
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
        )
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)

    if sub_disease:
        with open(
            os.path.join(logs_path, f"{name}_sub_disease.json"), "w"
        ) as json_file:
            json.dump(history.history, json_file)
    else:
        with open(os.path.join(logs_path, f"{name}_train_logs.json"), "w") as json_file:
            json.dump(history.history, json_file)


if __name__ == "__main__":
    """Main function to run the training of the model."""

    # Set the GPU to allocate only used memory at runtime.
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)

    # Args parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/ptb", help="Ptb-XL dataset location"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="imle_net",
        help="Select the model to train. (imle_net, mousavi, rajpurkar)",
    )
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size (Choose smaller batch size if available GPU memory is less)") 
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument(
        "--loggr", type=str2bool, default=False, help="Enable wandb logging"
    )
    parser.add_argument(
        "--sub",
        type=str2bool,
        default=False,
        help="Enable sub-diagnostic diseases of MI classification",
    )
    args = parser.parse_args()

    if args.model == "imle_net":
        from models.IMLENet import build_imle_net
        from configs.imle_config import Config

        model = build_imle_net(Config(), sub=args.sub)
    elif args.model == "mousavi":
        from models.mousavi import build_mousavi
        from configs.mousavi_config import Config

        model = build_mousavi(Config(), sub=args.sub)
    elif args.model == "rajpurkar":
        from models.rajpurkar import build_rajpurkar
        from configs.rajpurkar_config import params

        model = build_rajpurkar(sub=args.sub, **params)

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
        sub_disease=args.sub,
    )
