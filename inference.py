"""Inference and visualization script for the imle-net model.
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import logging
from utils.tf_utils import set_tf_loglevel

set_tf_loglevel(logging.ERROR)

import argparse, wfdb, os
import numpy as np
from joblib import load
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px

from preprocessing.preprocess import apply_scaler
from models.IMLENet import build_imle_net
from configs.imle_config import Config


def build_model(name: str = "imle_net") -> tf.keras.Model:
    """ Build the model and load the pretrained weights.

    Parameters
    ----------
    name: str, optional
        Name of the model, only supports imle-net
    """

    model = build_imle_net(Config())
    outputs = tf.keras.layers.Dense(3, activation='softmax')(model.layers[-2].output[0])
    model = tf.keras.models.Model(inputs = model.input, outputs = outputs)
    try:
        path_weights = os.path.join(os.getcwd(), "checkpoints", f"{name}_sub_diagnostic_weights.h5")
        model.load_weights(path_weights)
    except:
        raise Exception("Model weights file not found, please train the model on sub-diagnostic diseases of MI first.")     
    return model


def data_utils(dir: str) -> np.asarray:
    """ Preprocessing pipeline for a sample.

    Parameters
    ----------
    dir: str
        Path to the sample, should contain the name of sample without extensions.
    """

    sample_path = os.path.join(os.getcwd(), dir)
    data = np.array([wfdb.rdsamp(sample_path)[0]])
 
    scaler_path = os.path.join(os.getcwd(), 'data', 'standardization.bin')
    scaler = load(scaler_path)
    data = apply_scaler(data, scaler)
    data = np.transpose(data[..., np.newaxis], (0,2,1,3))
    return data


def build_scores(model: tf.keras.Model, data: np.asarray, config) -> None:
    """ Calculating the attention scores and visualization.

    Parameters
    ----------
    model: tf.keras.Model
        Model to be trained.
    data: np.asarray
        Sample data to be visualized.
    config:
        Configuration for the imle-net model.

    """

    scores_model = tf.keras.models.Model(inputs = model.input, outputs = [model.get_layer("beat_att").output, 
                                                                                        model.get_layer("rhythm_att").output,
                                                                                        model.get_layer("channel_att").output])
    beat, rhythm, channel = scores_model(data)
    beat = beat[1].numpy(); rhythm = rhythm[1].numpy(); channel = channel[1].numpy()

    # Beat scores
    lin = np.linspace(0, config.input_channels , num= config.beat_len)
    beat = beat.reshape(240, 13)
    beat_only = np.empty((240,config.beat_len))
    for i in range(beat.shape[0]):
        beat_only[i] = np.interp(lin, np.arange(13), beat[i])

    # Rhythm scores
    rhythm = rhythm.reshape(config.input_channels*int(config.signal_len / config.beat_len))
    
    # Channel scores
    channel = channel.flatten()

    # Beat scores using channel
    beat_channel = np.copy(beat_only.reshape(config.input_channels, config.beat_len * int(config.signal_len / config.beat_len)))     
    for i in range(config.input_channels):
        beat_channel[i] = beat_channel[i] * channel[i]

    beat_normalized = (beat_channel.flatten() - beat_channel.flatten().min(keepdims=True)) / (beat_channel.flatten().max( keepdims=True) - beat_channel.flatten().min(keepdims=True))
    beat_normalized = beat_normalized.reshape(config.input_channels, config.signal_len)
    v_min = np.min(beat_channel.flatten())
    v_max = np.max(beat_channel.flatten())
    
    ch_info = ['I',
           'II',
           'III',
           'AVR',
           'AVL',
           'AVF',
           'V1',
           'V2',
           'V3',
           'V4',
           'V5',
           'V6']
    results_filepath = os.path.join(os.getcwd(), "results")
    os.makedirs(results_filepath, exist_ok=True)

    fig, axs = plt.subplots(config.input_channels, figsize = (35, 25))
    data = data.squeeze()

    for i, (ax, ch) in enumerate(zip(axs, ch_info)):
        im = ax.scatter(np.arange(len(data[i])), data[i], cmap = 'Spectral', c= beat_normalized[i], vmin = v_min, vmax = v_max)
        ax.plot(data[i],color=(0.2, 0.68, 1))
        ax.set_yticks([])
        ax.set_title(ch, fontsize = 25)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([])
    plt.savefig(os.path.join(results_filepath, 'visualization.png'))

    fig = px.bar(channel, title = 'Channel Importance Scores')
    fig.update_xaxes(tickvals = np.arange(config.input_channels), ticktext = ch_info)
    fig.write_html(os.path.join(results_filepath, 'channel_visualization.html'))


if __name__ == "__main__":
    """Main function for visualization."""

    # Args parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default="data/sample/04827_lr", help="Sample file location, should contain the name of sample without extensions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="imle_net",
        help="Select the model to train. (only imle_net)",
    )
    args = parser.parse_args()

    config = Config()
    model = build_model(name=args.model)
    data = data_utils(args.dir)
    build_scores(model, data, config)
