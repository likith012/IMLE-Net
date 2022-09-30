"""An implementation of paper by Sajad Mousavi.

An ECG heartbeat detection model based on the paper by Sajad Mousavi. It's an encoder-decoder model based on LSTM.
More details can be found at  https://arxiv.org/pdf/1812.07421.pdf

Paper Name: Inter- and Intra-Patient ECG Heartbeat Classification For Arrhythmia Detection: A Sequence to Sequence Deep Learning Approach

"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Conv1D,
    Input,
    MaxPooling1D,
    Dense,
    Flatten,
    Bidirectional,
    LSTM,
    Permute,
)


def build_mousavi(config, sub=False) -> tf.keras.Model:
    """Builds Sajad Mousavi's model.

    Parameters
    ----------
    config: mousavi_config
        The configs for building the model.
    sub: bool, optional
        For sub-diagnostic diseases of MI. (default: False)

    Returns
    -------
    tf.keras.Model
        The keras sequential model.

    """

    inputs = Input(shape=(config.input_channels, config.signal_len, 1), batch_size=None)
    x = K.reshape(inputs, (-1, config.input_channels, config.signal_len))
    x = Permute((2, 1))(x)
    x = K.reshape(inputs, (-1, config.beat_len, config.input_channels))

    x = Conv1D(
        kernel_size=config.kernel_size,
        filters=config.filter_size[0],
        activation="relu",
        padding="same",
    )(x)
    x = MaxPooling1D(pool_size=config.pool_size)(x)
    x = Conv1D(
        kernel_size=config.kernel_size,
        filters=config.filter_size[1],
        activation="relu",
        padding="same",
    )(x)
    x = MaxPooling1D(pool_size=config.pool_size)(x)
    x = Conv1D(
        kernel_size=config.kernel_size,
        filters=config.filter_size[2],
        activation="relu",
        padding="same",
    )(x)

    x = Flatten()(x)
    x = K.reshape(x, (-1, 20, 1536))
    x = Bidirectional(LSTM(config.lstm_units, return_sequences=True))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(config.classes, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    if not sub:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.AUC(multi_label=True)],
        )
        model._name = "Mousavi"
        print(model.summary())

    return model
