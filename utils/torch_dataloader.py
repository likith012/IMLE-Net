"""Data Loader to load data as a torch Dataset object.

This file can also be imported as a module and contains the following functions:

    * DataGen - Generates a torch Dataset object.
    
"""

import math
from typing import Tuple

import numpy as np
import torch


class DataGen(torch.utils.data.Dataset):
    """Generates a torch Dataset object.

    Attributes
    ----------
    X: np.array
        Array of ECG signals.
    y: np.array
        Array of labels.
    batch_size: int, optional
        Batch size. (default: 32)

    """

    def __init__(self, X: np.array, y: np.array, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_x = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return torch.tensor(batch_x, dtype=torch.float32), torch.tensor(
            batch_y, dtype=torch.float32
        )
