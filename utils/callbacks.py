"""Custom callback for logging the metrics and saving the model.

This file can also be imported as a module and contains the following classes:

    * model_checkpoint - Custom callback for saving the model and logging the metrics.

"""
__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from utils.metrics import Metrics


class model_checkpoint(tf.keras.callbacks.Callback):
    """Custom callback for saving the model and logging the metrics.

    Arguments
    ---------
    savepath: str
        Path to the directory where the model will be saved.
    test_data: tf.keras.utils.Sequence
        Dataset object containing the test data.
    loggr: bool, optional
        Wandb object to log metrics. (default: False)
    monitor: str, optional
        Metric to monitor. (default: 'loss')
    name: str, optional
        Name of the model. (default: 'imle_net')
    sub: bool, optional
        For sub-diagnostic diseases of MI. (default: False)

    Methods
    -------
    on_epoch_end(epoch, logs = {})
        Saves the model and logs the metrics at the end of each epoch.

    """

    def __init__(
        self,
        savepath: str,
        test_data: tf.keras.utils.Sequence,
        loggr: bool = False,
        monitor: str = "loss",
        name: str = "imle_net",
        sub: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.savepath = savepath
        self.monitor = monitor
        self.test_data = test_data
        self.loggr = loggr
        self.name = name
        self.sub = sub
        self.best_score = 0.0

    def on_epoch_end(self, epoch: int, logs: dict = {}) -> None:
        """Saves the model and logs the metrics at the end of each epoch.

        Parameters
        ----------
        epoch: int
            Epoch number.
        logs: dict, optional
            Dictionary containing the metrics. (default: {})

        """

        test_len = len(self.test_data)
        score = []
        gt = []

        for i in range(test_len):
            X, y = self.test_data[i][0], self.test_data[i][1]
            temp_score = self.model.predict(X)
            score.append(temp_score)
            gt.append(y)

        score = np.concatenate(score, axis=0)
        gt = np.concatenate(gt, axis=0)
        roc_auc = roc_auc_score(gt, score, average="macro")
        _, accuracy = Metrics(gt, score)

        if self.sub:
            temp_path = f"{self.name}_sub_diagnostic_weights.h5" # Here we are saving the sub_diagnostic model weights when we run (train.py with --sub=True) for visualisation
        else:
            temp_path = f"{self.name}_weights.h5" # Here we are saving the model weights when we run train.py
        path = os.path.join(self.savepath, temp_path)

        if epoch > 5 and self.best_score < roc_auc:
            self.best_score = roc_auc
            self.model.save_weights(path)

        if self.loggr is not None:
            self.loggr.log({"train_loss": logs["loss"], "epoch": epoch})
            self.loggr.log(
                {"train_keras_auroc": logs.get(self.monitor), "epoch": epoch}
            )

            self.loggr.log({"val_loss": logs["val_loss"], "epoch": epoch})
            self.loggr.log({"val_keras_auroc": logs["val_auc"], "epoch": epoch})
            self.loggr.log({"val_roc_score": roc_auc, "epoch": epoch})
            self.loggr.log({"val_accuracy_score": accuracy, "epoch": epoch})

        logs["val_roc_auc"] = roc_auc
        logs["val_accuracy_score"] = accuracy

    def set_model(self, model: tf.keras.Model) -> None:
        """Sets the model to be saved."""
        self.model = model
