from tensorflow import keras


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
