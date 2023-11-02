"""This file contains the model and configuration for the model used."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import keras
from dataclasses_json import dataclass_json
from keras import layers as l


@dataclass_json
@dataclass(frozen=True)
class CNNConfig:
    """
    Configuration fon the `CNN` model

    args:
        filters: list of number of filters for each convolutional layer
        kernels: list of kernel size for each convolutional layer
        strides: list of stride size for each convolutional layer
        pool: list of `pool_size` in `MaxPool1D` for each convolutional layer, `None` to skip pooling
        padding: str of "same" or "valid", used in the `Conv1D` in all convolutional blocks
        fc_end: list of number of hidden layers in each `Dense` block
        dropout: input to `Dropout` after each convolutional block
    """

    filters: list[int]
    kernels: list[int]
    strides: list[int]
    pool: list[Optional[int]]
    padding: str
    fc_end: list[int]
    dropout: float


class ConvBlock(l.Layer):
    """
    Convolutional block consisting of `Conv1D`, `BatchNormalization`, optional `MaxPool1D` and `ReLU` activation

    args:
        filters: filters in `Conv1D`
        kernel_size: kernel_size in `Conv1D`
        stride: stride in `Conv1D`
        padding: padding in `Conv1D`
        pool: Optional[int] - optional `pool_size` in `MaxPool1D`. If None pooling is not done
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        stride: int,
        padding: str,
        pool: Optional[int],
    ) -> None:
        super().__init__()
        self.conv = l.Conv1D(filters, kernel_size, stride, padding)
        self.bn = l.BatchNormalization()
        if pool:
            self.pool = l.MaxPool1D(pool)  # , pool[1])
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
    """
    CNN Model consisting of:
    - N convolutional blocks (Conv1D, BatchNormalization, MaxPool1D, ReLU, Dropout)
    - GlobalMaxPool1D
    - M Fully Connected hidden layers
    - Fully Connected output layer

    args:
        config: `CNNConfig`, a dataclass containing all configuration options for the model.
    """

    def __init__(self, config: CNNConfig) -> None:
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
