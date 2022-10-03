"""An implementation of IMLE-Net: An Interpretable Multi-level Multi-channel Model for ECG Classification

Abstract: Early detection of cardiovascular diseases is crucial for effective treatment and an electrocardiogram (ECG)
        is pivotal for diagnosis. The accuracy of Deep Learning based methods for ECG signal classification has progressed
        in recent years to reach cardiologist-level performance. In clinical settings, a cardiologist makes a diagnosis based on
        the standard 12-channel ECG recording. Automatic analysis of ECG recordings from a multiple-channel perspective has not
        been given enough attention, so it is essential to analyze an ECG recording from a multiple-channel perspective. We propose a
        model that leverages the multiple-channel information available in the standard 12-channel ECG recordings and learns patterns
        at the beat, rhythm, and channel level. The experimental results show that our model achieved a macro-averaged ROC-AUC
        score of 0.9216, mean accuracy of 88.85% and a maximum F1 score of 0.8057 on the PTB-XL dataset. The attention
        visualization results from the interpretable model are compared against the cardiologist’s guidelines to validate the correctness
        and usability.

More details on the paper at https://ieeexplore.ieee.org/document/9658706

This file can also be imported as a module and contains the following functions:

    * attention - Feed-forward attention layer
    * residual_block - Implementation of a single Residual block
    * build_imle_net - Builds the IMLE-Net model

"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"

from typing import Tuple

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Conv1D,
    Input,
    LSTM,
    Activation,
    Dense,
    ReLU,
    BatchNormalization,
    Add,
    Bidirectional,
)


class attention(tf.keras.layers.Layer):
    """A class used to build the feed-forward attention layer.

    Attributes
    ----------
    return_sequences: bool, optional
        If False, returns the calculated attention weighted sum of an ECG signal. (default: False)
    dim: int, optional
        The dimension of the attention layer. (default: 64)

    Methods
    -------
    build(input_shape)
        Sets the weights for calculating the attention layer.
    call(x)
        Calculates the attention weights.
    get_config()
        Useful for serialization of the attention layer.

    """

    def __init__(self, return_sequences: bool = False, dim: int = 64, **kwargs) -> None:

        self.return_sequences = return_sequences
        self.dim = dim
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape: Tuple[int, int, int]) -> None:
        """Builds the attention layer.

        alpha = softmax(V.T * tanh(W.T * x + b))

        Parameters
        ----------
        W: tf.Tensor
            The weights of the attention layer.
        b: tf.Tensor
            The bias of the attention layer.
        V: tf.Tensor
            The secondary weights of the attention layer.

        """

        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], self.dim), initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], self.dim), initializer="zeros"
        )
        self.V = self.add_weight(name="Vatt", shape=(self.dim, 1), initializer="normal")
        super(attention, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Calculates the attention weights.

        Parameters
        ----------
        x: tf.Tensor
            The input tensor.

        Returns
        -------
        tf.Tensor
            The attention weighted sum of the input tensor.
        """

        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.dot(e, self.V)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output, a

        return K.sum(output, axis=1), a

    def get_config(self):
        """Returns the config of the attention layer. Useful for serialization."""

        base_config = super().get_config()
        config = {
            "return sequences": tf.keras.initializers.serialize(self.return_sequences),
            "att dim": tf.keras.initializers.serialize(self.dim),
        }
        return dict(list(base_config.items()) + list(config.items()))


def relu_bn(inputs: tf.Tensor) -> tf.Tensor:
    """ReLU activation followed by Batch Normalization.

    Parameters
    ----------
    inputs: tf.Tensor
        The input tensor.

    Returns
    -------
    tf.Tensor
        ReLU and Batch Normalization applied to the input tensor.
    """

    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(
    x: tf.Tensor, downsample: bool, filters: int, kernel_size: int = 8
) -> tf.Tensor:
    """[summary]

    Parameters
    ----------
    x: tf.Tensor
        The input tensor.
    downsample: bool
        If True, downsamples the input tensor.
    filters: int
        The number of filters in the 1D-convolutional layers.
    kernel_size: int, optional
        The kernel size of the 1D-convolutional layers. (default: 8)

    Returns
    -------
    tf.Tensor
        The output tensor of the residual block.
    """

    y = Conv1D(
        kernel_size=kernel_size,
        strides=(1 if not downsample else 2),
        filters=filters,
        padding="same",
    )(x)
    y = relu_bn(y)
    y = Conv1D(kernel_size=kernel_size, strides=1, filters=filters, padding="same")(y)

    if downsample:
        x = Conv1D(kernel_size=1, strides=2, filters=filters, padding="same")(x)

    out = Add()([x, y])
    out = relu_bn(out)
    return out


def build_imle_net(config, sub=False) -> tf.keras.Model:
    """Builds the IMLE-Net model.

    Parameters
    ----------
    config: imle_config
        The configs for building the model.
    sub: bool, optional
        For sub-diagnostic diseases of MI. (default: False)

    Returns
    -------
    tf.keras.Model
        The keras sequential model.

    """

    inputs = Input(shape=(config.input_channels, config.signal_len, 1), batch_size=None)

    # Beat Level
    x = K.reshape(inputs, (-1, config.beat_len, 1))
    x = Conv1D(
        filters=config.start_filters, kernel_size=config.kernel_size, padding="same"
    )(x)
    x = Activation("relu")(x)

    num_filters = config.start_filters
    for i in range(len(config.num_blocks_list)):
        num_blocks = config.num_blocks_list[i]
        for j in range(num_blocks):
            x = residual_block(x, downsample=(j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    x, _ = attention(name="beat_att")(x)

    # Rhythm level
    x = K.reshape(x, (-1, int(config.signal_len / config.beat_len), 128))
    x = Bidirectional(LSTM(config.lstm_units, return_sequences=True))(x)
    x, _ = attention(name="rhythm_att")(x)

    # Channel level
    x = K.reshape(x, (-1, config.input_channels, 128))
    x, _ = attention(name="channel_att")(x)
    outputs = Dense(config.classes, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    if not sub:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.AUC(multi_label=True)],
        )
        model._name = "IMLE-Net"
        print(model.summary())

    return model
