"""Configs for building the IMLE-Net model.
"""


class Config:
    """A class used for IMLE-Net configs."""

    def __init__(self) -> None:
        """
        Parameters
        ----------
        signal_len: int
            The length of the input ECG signal (Time in secs * Sampling rate).
        input_channels: int
            The number of input channels of an ECG signal.
        beat_len: int
            The length of the segmented ECG beat (Time in secs * Sampling rate).
        kernel_size: int
            The kernel size of the 1D-convolution kernel.
        num_blocks_list: List[int]
            The number of residual blocks in the model.
        lstm_units: int
            The number of units in the LSTM layer.
        start_filters: int
            The number of filters at the start of the 1D-convolution layer.
        classes: int
        The number of classes in the output layer.

        """

        self.signal_len = 1000
        self.input_channels = 12
        self.beat_len = 50
        self.kernel_size = 8
        self.num_blocks_list = [2, 2, 2]
        self.lstm_units = 64
        self.start_filters = 32
        self.classes = 5
