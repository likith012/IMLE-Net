# IMLE-Net: An Interpretable Multi-level Multi-channel Model for ECG Classification 
This repostiory contains code, results and dataset links for our paper titled ***IMLE-Net: An Interpretable Multi-level Multi-channel Model for ECG Classification***. 📝
>**Authors:** Likith Reddy, Vivek Talwar, Shanmukh Alle, Raju. S. Bapi, U. Deva Priyakumar.

>**More details on the paper can be found [here](https://ieeexplore.ieee.org/document/9658706).**

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
- Complete preprocessing pipeline, training and inference codes are available.
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

> 12-Lead ECG results



## Dataset ⚡



## Getting started 🥷

## Getting the weights :weight_lifting:

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

