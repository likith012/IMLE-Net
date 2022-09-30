"""Preprocessing pipeline for the dataset.

This file can is imported as a module and contains the following functions:

    * apply_scaler - Applies standard scaler to an ECG signal.
    * preprocess - Preprocesses the dataset.

"""
__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import os
from typing import Tuple

import numpy as np
import pandas as pd
import wfdb, ast
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


def apply_scaler(inputs: np.array, scaler: StandardScaler) -> np.array:
    """Applies standardization to each individual ECG signal.

    Parameters
    ----------
    inputs: np.array
        Array of ECG signals.
    scaler: StandardScaler
        Standard scaler object.

    Returns
    -------
    np.array
        Array of standardized ECG signals.

    """

    temp = []
    for x in inputs:
        x_shape = x.shape
        temp.append(scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    temp = np.array(temp)
    return temp


def preprocess(path: str = "data/ptb") -> Tuple[np.array]:
    """Preprocesses the dataset.

    Parameters
    ----------
    path: str
        Path to the dataset. (default: 'data/ptb')

    Returns
    -------
    tuple[np.array]
        Tuple of arrays containing train, valid and test data.

    """

    print("Loading dataset...", end="\n" * 2)

    path = os.path.join(os.getcwd(), Path(path))
    Y = pd.read_csv(os.path.join(path, "ptbxl_database.csv"), index_col="ecg_id")
    data = np.array([wfdb.rdsamp(os.path.join(path, f))[0] for f in Y.filename_lr])
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    agg_df = pd.read_csv(os.path.join(path, "scp_statements.csv"), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def agg(y_dic):
        temp = []

        for key in y_dic.keys():
            if key in agg_df.index:
                c = agg_df.loc[key].diagnostic_class
                if str(c) != "nan":
                    temp.append(c)
        return list(set(temp))

    Y["diagnostic_superclass"] = Y.scp_codes.apply(agg)
    Y["superdiagnostic_len"] = Y["diagnostic_superclass"].apply(lambda x: len(x))
    counts = pd.Series(np.concatenate(Y.diagnostic_superclass.values)).value_counts()
    Y["diagnostic_superclass"] = Y["diagnostic_superclass"].apply(
        lambda x: list(set(x).intersection(set(counts.index.values)))
    )

    X_data = data[Y["superdiagnostic_len"] >= 1]
    Y_data = Y[Y["superdiagnostic_len"] >= 1]

    print("Preprocessing dataset...", end="\n" * 2)

    mlb = MultiLabelBinarizer()
    mlb.fit(Y_data["diagnostic_superclass"])
    y = mlb.transform(Y_data["diagnostic_superclass"].values)

    # Stratified split
    X_train = X_data[Y_data.strat_fold < 9]
    y_train = y[Y_data.strat_fold < 9]

    X_val = X_data[Y_data.strat_fold == 9]
    y_val = y[Y_data.strat_fold == 9]

    X_test = X_data[Y_data.strat_fold == 10]
    y_test = y[Y_data.strat_fold == 10]

    del X_data, Y_data, y, data

    # Standardization
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    X_train_scale = apply_scaler(X_train, scaler)
    X_test_scale = apply_scaler(X_test, scaler)
    X_val_scale = apply_scaler(X_val, scaler)

    del X_train, X_test, X_val

    # Shuffling
    X_train_scale, y_train = shuffle(X_train_scale, y_train, random_state=42)

    return X_train_scale, y_train, X_test_scale, y_test, X_val_scale, y_val
