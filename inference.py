"""Inference and visualization script for the imle-net model .
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"

import argparse
import os
import numpy as np
from joblib import load
import wfdb

import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px

from preprocessing.preprocess import apply_scaler


from models.IMLENet import build_imle_net
from configs.imle_config import Config



def build_model(name="imle_net"):
    model = build_imle_net(Config())
    outputs = tf.keras.layers.Dense(3, activation='softmax')(model.layers[-2].output[0])
    model = tf.keras.models.Model(inputs = model.input, outputs = outputs)
    try:
        path_weights = os.path.join(os.getcwd(), "checkpoints", f"{name}_sub_diagnostic_weights.h5")
    except:
        print("Model weights file not found, please train the model on sub-diagnostic diseases of MI first.")
    else:
        model.load_weights(path_weights)
    return model

def data_utils(dir):
    sample_path = os.path.join(os.getcwd(), dir)
    data = np.array([wfdb.rdsamp(sample_path)[0]])
 
    scaler_path = os.path.join(os.getcwd(), 'data', 'standardization.bin')
    scaler = load(scaler_path)

    data = apply_scaler(data, scaler)
    data = np.transpose(data[..., np.newaxis], (0,2,1,3))
    return data

def build_scores(model, data):
    attention_layer = tf.keras.models.Model(inputs = model.input, outputs = [model.get_layer("beat_att").output, 
                                                                                        model.get_layer("rhythm_att").output,
                                                                                        model.get_layer("channel_att").output])
    beat, rhythm, channel = attention_layer(data)
    beat_att = np.asarray(beat[1]); rhythm_att = np.asarray(rhythm[1]); channel_att = np.asarray(channel[1])

    beat_size = 50
    lin = np.linspace(0, 12 , num= beat_size)
    beat_att = beat_att.reshape(240, 13)
    beat_only_att = np.empty((240,beat_size))

    for i in range(beat_att.shape[0]):
        beat_only_att[i] = np.interp(lin, np.arange(13), beat_att[i])


    ## Rhytm
    rhythm_att = rhythm_att.reshape(12*20)
    
    # Channel
    channel_att = channel_att.flatten()

    # Calculate Beat level using channel level

    beat_channel = np.copy(beat_only_att.reshape(12, 20*50))
        
    for i in range(12):
        beat_channel[i] = beat_channel[i] * channel_att[i]
        
    beat_channel_nor = (beat_channel.flatten() - beat_channel.flatten().min(keepdims=True)) / (beat_channel.flatten().max( keepdims=True) - beat_channel.flatten().min(keepdims=True))
    beat_channel_nor = beat_channel_nor.reshape(12, 1000)

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

    fig, axs = plt.subplots(12, figsize = (35, 25))
    x = np.arange(1000)

    v_min = np.min(beat_channel.flatten())
    v_max = np.max(beat_channel.flatten())


    for i, (ax, ch) in enumerate(zip(axs, ch_info)):
        im = ax.scatter(np.arange(len(data[:,:,i].squeeze())), data[:,:,i].squeeze(), cmap = 'hot_r', c= beat_channel_nor[i], vmin = v_min, vmax = v_max)
        # plt.colorbar(im, ax = ax)
        ax.plot(data[:,:,i].squeeze(),color=(0.2, 0.68, 1))
        ax.set_yticks([])

        ax.set_title(ch, fontsize = 25)

    fig.tight_layout()

    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([])

    plt.show()  
    # plt.savefig('visualize.jpeg')



    fig = px.bar(channel_att, title = 'Channel Importance Scores')

    fig.update_xaxes(tickvals = np.arange(12), ticktext = ch_info)

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

    model = build_model(name=args.model)
    data = data_utils(args.dir)
    build_scores(model, data)
