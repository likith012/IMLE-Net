"""An implementation of Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks, Rajpurkar et al.

More details on the paper at https://arxiv.org/abs/1707.01836

This file can also be imported as a module and contains the following functions:

    * build_network: Builds the network according to the parameters specified in the parameters dictionary.
    * add_compile: Compiles the model according to the parameters specified in the parameters dictionary.
    * add_output_layer: Adds the output layer to the network.
    * resnet_block: Implements a resnet block to the network.
    * add_resnet_layers: Adds the resnet layers to the network.

"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Add,
    Lambda,
    MaxPooling1D,
    Conv1D,
    BatchNormalization,
    Activation,
    Dropout,
    Input,
    Permute,
)


def _bn_relu(layer: tf.Tensor, dropout: float = 0.0, **params) -> tf.Tensor:
    """Helper to build a BN -> relu block with dropout

    Parameters
    ----------
    layer: tf.Tensor
        The input layer to the block.
    dropout: float, optional
        The dropout rate.

    Returns
    -------
    tf.Tensor
        Applies batch normalization and relu activation to the input layer.

    """

    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)
    if dropout > 0:
        layer = Dropout(params["conv_dropout"])(layer)
    return layer


def add_conv_weight(
    layer: tf.Tensor,
    filter_length: int,
    num_filters: int,
    subsample_length: int = 1,
    **params
) -> tf.Tensor:
    """Helper to build a convolutional layer with a given filter length and number of filters.

    Parameters
    ----------
    layer: tf.Tensor
        The input layer to the block.
    filter_length: int
        The length of the filters.
    num_filters: int
        The number of filters to the 1D convolution.
    subsample_length: int, optional
        The length of the the strides to the 1D convolution.

    Returns
    -------
    tf.Tensor
        Apply 1D convolution to the input layer.

    """
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding="same",
        kernel_initializer=params["conv_init"],
    )(layer)
    return layer


def add_conv_layers(layer: tf.Tensor, **params) -> tf.Tensor:
    """Helper to build the convolutional layers of the network.

    Parameters
    ----------
    layer: tf.Tensor
        The input layer to the network.

    Returns
    -------
    tf.Tensor
        Applies the convolutional layers to the input layer.

    """

    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            params["conv_num_filters_start"],
            subsample_length=subsample_length,
            **params
        )
        layer = _bn_relu(layer, **params)
    return layer


def resnet_block(
    layer: tf.Tensor,
    num_filters: int,
    subsample_length: int,
    block_index: int,
    **params
) -> tf.Tensor:
    """Implements a resnet block to the network.

    Parameters
    ----------
    layer: tf.Tensor
        The input layer to the block.
    num_filters: int
        The number of filters to the 1D convolution.
    subsample_length: int
        The length of the the strides to the 1D convolution.
    block_index: int
        The index of the resnet block.

    Returns
    -------
    tf.Tensor
        Applies the resnet block to the input layer.

    """

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length, padding="same")(layer)
    zero_pad = (
        block_index % params["conv_increase_channels_at"]
    ) == 0 and block_index > 0

    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer, dropout=params["conv_dropout"] if i > 0 else 0, **params
            )
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params
        )
    layer = Add()([shortcut, layer])
    return layer


def get_num_filters_at_index(index: int, num_start_filters: int, **params) -> int:
    return 2 ** int(index / params["conv_increase_channels_at"]) * num_start_filters


def add_resnet_layers(layer: tf.Tensor, **params) -> tf.Tensor:
    """Adds the resnet layers to the network.

    Parameters
    ----------
    layer: tf.Tensor
        The input layer to the block.

    Returns
    -------
    tf.Tensor
        Applies the resnet layers to the input layer.

    """

    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params
    )
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params
        )
        layer = resnet_block(layer, num_filters, subsample_length, index, **params)
    layer = _bn_relu(layer, **params)
    return layer


def add_output_layer(layer: tf.Tensor, **params) -> tf.Tensor:
    """Adds the output layer to the network."""
    layer = Flatten()(layer)
    layer = Dense(5, activation="sigmoid")(layer)
    return layer


def add_compile(model: tf.keras.Model, **params) -> tf.keras.Model:
    """Compiles the model with the given parameters.

    Parameters
    ----------
    model: tf.keras.Model
        The model to compile.

    Returns
    -------
    tf.keras.Model
        Compiles the model with loss and optimizer.

    """

    optimizer = Adam(lr=params["learning_rate"], clipnorm=params.get("clipnorm", 1))
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizer,
        metrics=["accuracy", tf.keras.metrics.AUC(multi_label=True)],
    )


def build_rajpurkar(sub=False, **params) -> tf.keras.Model:
    """Builds the network with the given parameters.

    Parameters
    ----------
    sub: bool, optional
        For sub-diagnostic diseases of MI. (default: False)
    params: dict
        The parameters to build the network.

    Returns
    -------
    tf.keras.Model
        Returns the final model.

    """

    signal_len = 1000
    input_channels = 12

    inputs = Input(
        shape=(input_channels, signal_len, 1),
        dtype="float32",
        name="inputs",
        batch_size=None,
    )
    x = K.reshape(inputs, (-1, input_channels, signal_len))
    x = Permute((2, 1))(x)

    if params.get("is_regular_conv", False):
        layer = add_conv_layers(x, **params)
    else:
        layer = add_resnet_layers(x, **params)

    output = add_output_layer(layer, **params)
    model = tf.keras.models.Model(inputs=[inputs], outputs=[output])

    if not sub:
        if params.get("compile", True):
            add_compile(model, **params)
        model._name = "Rajpurkar"
        print(model.summary())

    return model
