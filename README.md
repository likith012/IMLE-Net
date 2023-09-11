[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![GitHub pull requests](https://img.shields.io/github/issues-pr/likith012/IMLE-Net)
![GitHub issues](https://img.shields.io/github/issues/likith012/IMLE-Net)

# IMLE-Net: An Interpretable Multi-level Multi-channel Model for ECG Classification 
This repository contains code, results and dataset links for our ***IEEE:SMC 2021*** oral paper titled ***IMLE-Net: An Interpretable Multi-level Multi-channel Model for ECG Classification***. 📝
>**Authors:** Likith Reddy, Vivek Talwar, Shanmukh Alle, Raju. S. Bapi, U. Deva Priyakumar.

>**More details on the paper can be found [here](https://ieeexplore.ieee.org/document/9658706). Also available on [arxiv](https://arxiv.org/abs/2204.05116).**

>**Raise an issue for any query regarding the code, paper or for any support.**

>**Tested and works with the latest tensorflow and torch versions at the time of this commit.**

## Table of contents
- [Introduction](https://github.com/likith012/IMLE-Net/edit/main/README.md#introduction-)
- [Highlights](https://github.com/likith012/IMLE-Net/edit/main/README.md#features-)
- [Results](https://github.com/likith012/IMLE-Net/edit/main/README.md#results)
- [Dataset](https://github.com/likith012/IMLE-Net/edit/main/README.md#organization-office)
- [Getting started](https://github.com/likith012/IMLE-Net/edit/main/README.md#getting-started-)
- [Getting the weights](https://github.com/likith012/IMLE-Net/edit/main/README.md#getting-the-weights-weight_lifting)
- [License and Citation](https://github.com/likith012/IMLE-Net/edit/main/README.md#license-and-citation-)

## Introduction 🔥

>Early detection of cardiovascular diseases is crucial for effective treatment and an electrocardiogram (ECG)
is pivotal for diagnosis. The accuracy of Deep Learning
based methods for ECG signal classification has progressed
in recent years to reach cardiologist-level performance. In
clinical settings, a cardiologist makes a diagnosis based on
the standard 12-channel ECG recording. Automatic analysis of
ECG recordings from a multiple-channel perspective has not
been given enough attention, so it is essential to analyze an ECG
recording from a multiple-channel perspective. We propose a
model that leverages the multiple-channel information available
in the standard 12-channel ECG recordings and learns patterns
at the beat, rhythm, and channel level. The experimental results
show that our model achieved a macro-averaged ROC-AUC
score of 0.9216, mean accuracy of 88.85% and a maximum
F1 score of 0.8057 on the PTB-XL dataset. The attention
visualization results from the interpretable model are compared
against the cardiologist’s guidelines to validate the correctness
and usability.

## Highlights ✨

- A model that learns patterns at the beat, rhythm, and channel level with high accuracy💯.
- An interpretable model that gives an explainability at the beat, rhythm and  channel level💥.
- Complete preprocessing pipeline, training and inference codes are provided.
- Training weights are available to try out the model.

## Results :man_dancing:

> Performance metrics

|          | Macro ROC-AUC | Mean Accuracy | Max. F1-score |
| -------- | ------------- | ------------- | ------------- |
| Resnet101 | 0.8952 | 86.78 | 0.7558 |
| Mousavi et al.| 0.8654 | 84.19 | 0.7315 | 
| ECGNet | 0.9101 | 87.35 | 0.7712 |
| Rajpurkar et al. | 0.9155 | 87.91 | 0.7895 |
| **IMLE-Net**| **0.9216** | **88.85** | **0.8057** |



> Visualization of normalized attention scores with red having a higher attention score and yellow having a lower attention score for a 12-lead ECG signal.

<img src="/images/viz_nor_final.png" width="800">

> Channel Importance scores for the same 12-lead ECG signal.

<img src="/images/graph.png" width="400">


## Dataset ⚡

#### Download :hourglass:

The `PTB-XL` dataset can be downloaded from the [Physionet website](https://physionet.org/content/ptb-xl/1.0.1/).

#### Getting started :ninja:

* To prepare the dataset, `cd IMLE-Net`
* Download the dataset using the terminal `wget -r -N -c -np -nH --cut-dirs 4 -O data/ptb.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip`
* Unzip and rename the dataset, `unzip data/ptb.zip -d data/ && mv data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2 data/ptb && rm data/ptb.zip`

>**More details on the dataset can be found [here](https://github.com/likith012/IMLE-Net/blob/main/data/download.md).**

## Getting started 🥷

#### Setting up the environment
- All the development work is done on `Python 3.7`
- Install necessary dependencies using `requirements.txt` file. Run `pip install -r requirements.txt` in terminal
- Alternatively, set up environment and train the model using `Dockerfile`. Run `docker build -f Dockerfile -t <image_name> .`

#### What each file does
- `train.py` trains a particular model from scratch
- `preprocessing` contains the preprocessing scripts
- `models` contains scripts for each model
- `utils` contains utilities for `dataloader`, `callbacks` and `metrics`

#### Training the model
- The models are implemented in either `tensorflow` or `torch`
- Models implemented in `tensorflow` are `imle_net`, `mousavi` and `rajpurkar`
- Models implemented in `torch` are `ecgnet` and `resnet101`
- To log the training and validation metrics using `wandb` tool, set `--loggr` to `True`
- To train a particular model from scratch, `cd IMLE-Net`
- To train a model on sub-diseases of MI, set `sub` to `True`
- To run `tensorflow` models, `python train.py --model imle_net --batchsize 32 --epochs 60 --loggr False`
- To run `torch` models, `python torch_train.py --model ecgnet --batchsize 32 --epochs 60 --loggr False`

#### Testing the model
- To test a model on `test dataset` after training the model, `cd IMLE-Net`
- For `tensorflow` models, `python test.py --model imle_net --batchsize 32`
- For `torch` models, `python torch_test.py --model ecgnet --batchsize 32`

#### Reimplementation and visualization
- It's a three step process, first train the model on main dataset with `sub` set to `False`.
- Second, train the model on sub-diseases of MI with `sub` set to `True`.
- Third, for inference and visualization run, `python inference.py --dir filepath`

#### Logs and checkpoints
- The logs are saved in `logs/` directory.
- The model checkpoints are saved in `checkpoints/` directory.
- The visualizations are saved in `results/` directory.

## Getting the weights :weight_lifting:

> Download the weights for several models trained on the PTB-XL dataset.

| Name | Author's | Community |
| :----: | :----------: | :---------: |
| Mousavi et al.| [link](https://drive.google.com/file/d/13nUC_9mlSdw-I_HfFai4k8k9bgOruQ-x/view?usp=sharing) | [link](https://drive.google.com/file/d/133rzsq6VJvW5BDa5IR7_CNDO_u39UtIa/view?usp=sharing) |
| ECGNet | [link](https://drive.google.com/file/d/1k0cgZBKQmkeVwu879NAtV-hDfLzCRCYJ/view?usp=sharing) | [link](https://drive.google.com/file/d/1Vx5oHilGxrxEJfUnLfg--Ow9AjyaUQu1/view?usp=sharing) |
| Rajpurkar et al. | [link](https://drive.google.com/file/d/18GZMDBAE2mHmQy8aXwD6cZoPjIAcavWX/view?usp=sharing) | [link](https://drive.google.com/file/d/16lJ8ICAvXzkyuGChesTMTsCN55GYI8Oj/view?usp=sharing) |
| **IMLE-Net**| [link](https://drive.google.com/file/d/1-ZJSEr_NtbLXWWx5otXT5ItE5p-Wc0HN/view?usp=sharing) | [link](https://drive.google.com/file/d/1ZF4nGjA-qjOb1e0gOGrVKrCACNDwLNn-/view?usp=sharing) |

## License and Citation 📰
The software is licensed under the Apache License 2.0. Please cite the following paper if you have used this code:
```
@INPROCEEDINGS{9658706,  
author={Reddy, Likith and Talwar, Vivek and Alle, Shanmukh and Bapi, Raju. S. and Priyakumar, U. Deva},  
booktitle={2021 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},   
title={IMLE-Net: An Interpretable Multi-level Multi-channel Model for ECG Classification},   
year={2021},  
pages={1068-1074}, 
doi={10.1109/SMC52423.2021.9658706}}
```

