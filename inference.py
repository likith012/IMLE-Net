"""Inference and visualization script for the imle-net model.
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import argparse, os, wfdb
import numpy as np
from joblib import load

import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px

from preprocessing.preprocess import apply_scaler
from models.IMLENet import build_imle_net
from configs.imle_config import Config


def build_model(name="imle_net"):
    """ Build the model and load the pretrained weights.

    Parameters
    ----------
    model: tf.keras.Model
        Model to be trained.
    path: str, optional
        Path to the directory containing the data. (default: 'data/ptb')
    batch_size: int, optional
        Batch size. (default: 32)
    name: str, optional
        Name of the model. (default: 'imle_net')

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


def data_utils(dir):
    """Testing the model and logging metrics.

    Parameters
    ----------
    model: tf.keras.Model
        Model to be trained.
    path: str, optional
        Path to the directory containing the data. (default: 'data/ptb')
    batch_size: int, optional
        Batch size. (default: 32)
    name: str, optional
        Name of the model. (default: 'imle_net')

    """

    sample_path = os.path.join(os.getcwd(), dir)
    data = np.array([wfdb.rdsamp(sample_path)[0]])
 
    scaler_path = os.path.join(os.getcwd(), 'data', 'standardization.bin')
    scaler = load(scaler_path)

    data = apply_scaler(data, scaler)
    data = np.transpose(data[..., np.newaxis], (0,2,1,3))
    return data


def build_scores(model, data, config):
    """Testing the model and logging metrics.

    Parameters
    ----------
    model: tf.keras.Model
        Model to be trained.
    path: str, optional
        Path to the directory containing the data. (default: 'data/ptb')
    batch_size: int, optional
        Batch size. (default: 32)
    name: str, optional
        Name of the model. (default: 'imle_net')

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
    rhythm = rhythm.reshape(config.input_channels*(config.signal_len / config.beat_len))
    
    # Channel scores
    channel = channel.flatten()

    # Beat scores using channel
    beat_channel = np.copy(beat_only.reshape(config.input_channels, (config.signal_len / config.beat_len)))
        
    for i in range(config.input_channels):
        beat_channel[i] = beat_channel[i] * channel[i]
        
    beat_channel_nor = (beat_channel.flatten() - beat_channel.flatten().min(keepdims=True)) / (beat_channel.flatten().max( keepdims=True) - beat_channel.flatten().min(keepdims=True))
    beat_channel_nor = beat_channel_nor.reshape(config.input_channels, config.signal_len)

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

    v_min = np.min(beat_channel.flatten())
    v_max = np.max(beat_channel.flatten())

    fig, axs = plt.subplots(config.input_channels, figsize = (35, 25))

    for i, (ax, ch) in enumerate(zip(axs, ch_info)):
        im = ax.scatter(np.arange(len(data[:,:,i].squeeze())), data[:,:,i].squeeze(), cmap = 'hot_r', c= beat_channel_nor[i], vmin = v_min, vmax = v_max)
        # plt.colorbar(im, ax = ax)
        ax.plot(data[:,:,i].squeeze(),color=(0.2, 0.68, 1))
        ax.set_yticks([])
        ax.set_title(ch, fontsize = 25)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([])

    results_filepath = os.path.join(os.getcwd(), "results")
    os.makedirs(results_filepath, exist_ok=True)

    fig = px.bar(channel, title = 'Channel Importance Scores')
    fig.update_xaxes(tickvals = np.arange(config.input_channels), ticktext = ch_info)
    fig.show()


if __name__ == "__main__":
    """Main function to test the trained model."""

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
