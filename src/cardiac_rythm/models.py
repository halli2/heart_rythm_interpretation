from dataclasses import dataclass
from typing import Optional

import keras
from dataclasses_json import dataclass_json
from keras import layers as l


@dataclass_json
@dataclass(frozen=True)
class CNNConfig:
    filters: list[int]
    kernels: list[int]
    strides: list[int]
    pool: list[Optional[tuple[int, int]]]  # list of pools or none
    padding: str  # 'same' Or 'valid'
    fc_end: list[int]
    dropout: float


class ConvBlock(l.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        stride: int,
        padding: str,
        pool: tuple[int, int],
    ):
        super().__init__()
        self.conv = l.Conv1D(filters, kernel_size, stride, padding)
        self.bn = l.BatchNormalization()
        if pool:
            self.pool = l.MaxPool1D(pool[0])  # , pool[1])
        else:
            self.pool = None
        self.activation = l.ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.pool is not None:
            x = self.pool(x)
        return self.activation(x)


class CNN(keras.Model):
    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config

        self.cnn_blocks = []
        settings = zip(config.filters, config.kernels, config.strides, config.pool)
        for filt, kernel, stride, pool in settings:
            self.cnn_blocks.append(ConvBlock(filt, kernel, stride, config.padding, pool))
        self.dropout = l.Dropout(config.dropout)
        # from Krasteva et al
        self.gmp = l.GlobalMaxPool1D()
        self.fc_blocks = []
        for fc in config.fc_end:
            self.fc_blocks.append(l.Dense(fc, activation="relu"))

        self.fc_out = l.Dense(5, activation="softmax")

    def call(self, x):
        for block in self.cnn_blocks:
            x = block(x)
            x = self.dropout(x)
        x = self.gmp(x)
        for block in self.fc_blocks:
            x = block(x)
        x = self.fc_out(x)
        return x
