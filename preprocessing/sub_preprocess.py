"""Preprocessing pipeline for the sub-diagnostic diseases of Myocardial Infarction (MI).

This file is imported as a module and contains the following functions:

    * preprocess_sub_disease - Preprocesses the sub-diagnostic diseases of MI.

"""
__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import os
from typing import Tuple

import numpy as np
import pandas as pd
import wfdb, ast, warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

from .preprocess import apply_scaler

warnings.filterwarnings("ignore")


def preprocess_sub_disease(path: str = "data/ptb") -> Tuple[np.array]:
    """Preprocess the sub-diagnostic diseases of MI, namely, Anteroseptal Myocardial Infarction (ASMI) and Inferior Myocardial Infarction (IMI)
        along with normal (NORM) ECG recordings.

    Parameters
    ----------
    path: str
        Path to the dataset. (default: 'data/ptb')

    Returns
    -------
    tuple[np.array]
        Tuple of arrays containing train and test data.

    """

    print("Loading dataset...", end="\n" * 2)

    path = os.path.join(os.getcwd(), Path(path))
    Y = pd.read_csv(os.path.join(path, "ptbxl_database.csv"), index_col="ecg_id")
    data = np.array([wfdb.rdsamp(os.path.join(path, f))[0] for f in Y.filename_lr])
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    agg_df = pd.read_csv(os.path.join(path, "scp_statements.csv"), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def MI_agg(y_dic):
        temp = []

        for key in y_dic.keys():
            if y_dic[key] in [100, 80, 0]:
                if key in agg_df.index:
                    if key in ["ASMI", "IMI"]:
                        temp.append(key)
        return list(set(temp))

    Y["diagnostic_subclass"] = Y.scp_codes.apply(MI_agg)
    Y["subdiagnostic_len"] = Y["diagnostic_subclass"].apply(lambda x: len(x))

    # MI sub-diseases ASMI and IMI
    x1 = data[Y["subdiagnostic_len"] == 1]
    y1 = Y[Y["subdiagnostic_len"] == 1]

    def norm_agg(y_dic):
        for key in y_dic.keys():
            if y_dic[key] in [100]:
                if key == "NORM":
                    return "NORM"

    N = Y.copy()
    N["diagnostic_subclass"] = Y.scp_codes.apply(norm_agg)

    ## Normal class
    x2 = data[N["diagnostic_subclass"] == "NORM"]
    y2 = N[N["diagnostic_subclass"] == "NORM"]

    # Train and test splits
    x1_train = x1[y1.strat_fold <= 8]
    y1_train = y1[y1.strat_fold <= 8]

    x1_test = x1[y1.strat_fold > 8]
    y1_test = y1[y1.strat_fold > 8]

    x2_train = x2[y2.strat_fold <= 2][:900]
    y2_train = y2[y2.strat_fold <= 2][:900]

    x2_test = x2[y2.strat_fold == 3][:200]
    y2_test = y2[y2.strat_fold == 3][:200]

    X_train = np.concatenate((x1_train, x2_train), axis=0)
    X_test = np.concatenate((x1_test, x2_test), axis=0)

    y1_train.diagnostic_subclass = y1_train.diagnostic_subclass.apply(lambda x: x[0])
    y1_test.diagnostic_subclass = y1_test.diagnostic_subclass.apply(lambda x: x[0])
    y_train = np.concatenate(
        (y1_train.diagnostic_subclass.values, y2_train.diagnostic_subclass.values),
        axis=0,
    )
    y_test = np.concatenate(
        (y1_test.diagnostic_subclass.values, y2_test.diagnostic_subclass.values), axis=0
    )

    del data, x1, x2, y1, y2, x1_train, y1_train, y2_train, x2_train

    print("Preprocessing dataset...", end="\n" * 2)

    le = LabelEncoder()
    y_train = to_categorical(le.fit_transform(y_train))
    y_test = to_categorical(le.transform(y_test))

    # Standardization
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    X_train = apply_scaler(X_train, scaler)
    X_test = apply_scaler(X_test, scaler)

    # Shuffling
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    return X_train, y_train, X_test, y_test
