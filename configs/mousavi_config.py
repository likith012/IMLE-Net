"""Configs for building the Mousavi model.
"""


class Config:
    """A class used for mousavi model configs"""

    def __init__(self) -> None:
        """
        Parameters
        ----------
        signal_len: int
            The length of the input ECG signal(Time in secs * Sampling rate).
        input_channels: int
            The number of input channels of an ECG signal.
        beat_len: int
            The length of the segmented ECG beat(Time in secs * Sampling rate).
        kernel_size: int
            The kernel size of the 1D-convolution kernel.
        pool_size: int
            The pool size of the 1D-max-pooling kernel.
        lstm_units: int
            The number of units in the LSTM layer.
        filters: List[int]
            The number of filters in each 1D-convolution layer.
        classes: int
            The number of classes in the output layer.

        """

        self.signal_len = 1000
        self.input_channels = 12
        self.beat_len = 50
        self.kernel_size = 2
        self.pool_size = 2
        self.lstm_units = 128
        self.filter_size = [32, 64, 128]
        self.classes = 5
