from dataclasses import dataclass
from enum import Enum

import keras
from dataclasses_json import dataclass_json
from keras import layers as l


# TODO: Use THIS
class Pool(Enum):
    NEVER = 0
    EACH = 1
    AFTER = 2


@dataclass_json
@dataclass(frozen=True)
class CNNConfig:
    filters: list[int]  # [32, 64, 128, 256, 256]
    kernel_size: int | list[int]  # [3, 3, 1, 1, 1]
    stride: int = 1
    pool: tuple[int, int] = (3, 2)
    pool_each_conv: bool = False
    pool_after: bool = False
    padding: str = "same"  # Or valid
    fc1: int = 512
    fc2: int = 1024


class ConvBlock(l.Layer):
    def __init__(self, filters, kernel_size, stride, padding, pool, *, enable_pool: bool):
        super().__init__()
        self.conv = l.Conv1D(filters, kernel_size, stride, padding)
        self.bn = l.BatchNormalization()
        if enable_pool:
            self.pool = l.MaxPool1D(pool[0], pool[1])
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
        enable_pooling = config.pool_each_conv
        for i, filt in enumerate(config.filters):
            if i + 1 == len(config.filters) and config.pool_after:
                enable_pooling = True
            if config.kernel_size is int:
                kernel_size = config.kernel_size
            else:
                kernel_size = config.kernel_size[i]
            self.cnn_blocks.append(
                ConvBlock(
                    filt,
                    kernel_size,
                    config.stride,
                    config.padding,
                    config.pool,
                    enable_pool=enable_pooling,
                )
            )
        self.dropout = l.Dropout(0.5)
        self.flatten = l.Flatten()
        self.fc1 = l.Dense(config.fc1, activation="relu")
        self.fc2 = l.Dense(config.fc2, activation="relu")
        self.fc_out = l.Dense(5, activation="softmax")

    def call(self, x):
        for block in self.cnn_blocks:
            x = block(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        return x


# TODO: Finn ut hvorfor layersene er brukt
# Hva er PSP? Padding stride pooling?
def CNN_PSP(input_shape: tuple[int, int], pad: bool, stride: bool, pool: int) -> keras.Sequential:
    """Four-layer CNN experiment model - Padding and stride"""
    nfilters = [32] * 4
    kernel_size = [1] * 4
    model = keras.Sequential()
    model.add(keras.Input(input_shape))

    if pad and stride:
        for i, filters in enumerate(nfilters):
            model.add(
                keras.layers.Conv1D(
                    filters,
                    kernel_size[i],
                    strides=2,
                    padding="same",
                    activation="relu",
                )
            )
            model.add(keras.layers.BatchNormalization())
    elif pad and pool == 1:
        for i, filters in enumerate(nfilters):
            model.add(
                keras.layers.Conv1D(
                    filters,
                    kernel_size[i],
                    padding="same",
                    activation="relu",
                )
            )
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.MaxPool1D(3, 2))
    elif pad:
        for i, filters in enumerate(nfilters):
            model.add(
                keras.laters.Conv1D(
                    filters,
                    kernel_size[i],
                    padding="same",
                    activation="relu",
                )
            )
            model.add(keras.layers.BatchNormalization())
    else:
        for i, filters in enumerate(nfilters):
            model.add(
                keras.layers.Conv1D(
                    filters,
                    kernel_size[i],
                    padding="valid",
                    activation="relu",
                )
            )
            model.add(keras.layers.BatchNormalization())

    if pool == 2:
        model.add(keras.layers.MaxPool1D(2, 1))

    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten(name="flatten"))

    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dense(5, activation="softmax"))

    return model


def CNN_5(input_shape: tuple[int, int], increasing: bool) -> keras.Sequential:
    nfilters = [32] * 5
    kernel_size = [1] * 5
    if increasing:
        nfilters = [32, 64, 128, 264, 264]
        kernel_size = [3, 3, 1, 1, 1]

    model = keras.Sequential()
    model.add(keras.Input(input_shape))

    for i, filters in enumerate(nfilters):
        model.add(
            keras.layers.Conv1D(
                filters,
                kernel_size[i],
                padding="same",
                activation="relu",
            )
        )
        model.add(
            keras.layers.MaxPooling1D(
                pool_size=4,
                strides=3,
                padding="valid",
            )
        )
        model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten(name="flatten"))

    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dense(5, activation="softmax"))
    return model


if __name__ == "__main__":
    model = CNN_5((1000, 1), True)
    model.summary()
