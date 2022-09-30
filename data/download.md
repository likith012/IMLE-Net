## Download :hourglass:

The `PTB-XL` dataset can be downloaded from the [Physionet website](https://physionet.org/content/ptb-xl/1.0.1/).

## Getting started :ninja:
* To prepare the dataset, `cd IMLE-Net`
* Download the dataset using the terminal `wget -r -N -c -np -nH --cut-dirs 4 -O data/ptb.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip`
* Unzip and rename the dataset, `unzip data/ptb.zip -d data/ && mv data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2 data/ptb && rm data/ptb.zip`

## Description :information_desk_person:

The dataset used is the PTB-XL dataset which is
the largest openly available dataset that provides clinical 12
channel ECG waveforms. It comprises 21837 ECG records
from 18885 patients of 10 seconds length which follow
the standard set of channels (I, II, III, aVL, aVR, aVF,
V1–V6). The dataset is balanced concerning sex with 52%
male and 48% female and covers age ranging from 0 to 95
years. The dataset covers a wide range of pathologies with
many different co-occurring diseases. The ECG waveform
records are annotated by two certified cardiologists. Each
ECG record has labels assigned out of a set of 71 different
statements conforming to the Standard communications pro-
tocol for computer assisted electrocardiography (SCP-ECG)
standard. The ECG waveform was originally recorded at a
sampling rate of 400 Hz and downsampled to 100 Hz. All
the experiments in our work were performed using the 100
Hz sampling rate.

## Data organization :office:

```
ptbxl
├── ptbxl_database.csv
├── scp_statements.csv
├── records100
│   ├── 00000
│   │   ├── 00001_lr.dat
│   │   ├── 00001_lr.hea
│   │   ├── ...
│   │   ├── 00999_lr.dat
│   │   └── 00999_lr.hea
│   ├── ...
│   └── 21000
│        ├── 21001_lr.dat
│        ├── 21001_lr.hea
│        ├── ...
│        ├── 21837_lr.dat
│        └── 21837_lr.hea
└── records500
   ├── 00000
   │     ├── 00001_hr.dat
   │     ├── 00001_hr.hea
   │     ├── ...
   │     ├── 00999_hr.dat
   │     └── 00999_hr.hea
   ├── ...
   └── 21000
          ├── 21001_hr.dat
          ├── 21001_hr.hea
          ├── ...
          ├── 21837_hr.dat
          └── 21837_hr.hea
```
