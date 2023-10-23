import os
import pickle
from dataclasses import dataclass
from datetime import datetime

import keras
import numpy as np
import pandas as pd
import preprocessing
import tensorflow as tf
import visualize
from dataclasses_json import dataclass_json
from models import CNN, CNNConfig
from sklearn import model_selection

MODEL = "CNN_TODO"
# # PSP Settings
# PAD = True
# STRIDE = False
# POOL = 1
# # CNN_5 Settings
# INCREASING = True


@dataclass_json
@dataclass(frozen=True)
class FitSettings:
    epochs: int = 100
    batch_size: int = 32
    folds: int = 10
    normalize_data_length: bool = False  # Normalize, or weigh


def fit(file_path: str, fit_settings: FitSettings, model_config: CNNConfig) -> None:
    """Loads the data, model and fits it."""
    # Set random seed for reproducibility
    rng = np.random.RandomState(0)
    # Load data and create dataframe
    df = preprocessing.load_data(file_path)

    if fit_settings.normalize_data_length:
        df = preprocessing.normalize_data_length(df)

    # TODO: Initialize to 0:1,1:1,2:1 etc..
    class_weights = {}
    total = len(df["c_label"])
    class_counts = df["c_label"].value_counts().sort_index().values
    for k, v in enumerate(class_counts):
        class_weights[k] = 1 - (v / total)

    x = np.stack(df["s_ecg"].to_numpy())
    x = x.reshape((*x.shape, 1))
    y = df["c_label"].to_numpy()
    y = y - 1  # 0-4 instead of 1-5
    # Should we stratify based on patients?
    # groups = np.stack(df_data["patID"])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=rng,
        stratify=y,
    )

    skf = model_selection.StratifiedKFold(n_splits=fit_settings.folds, random_state=rng, shuffle=True)

    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    result_dir = f"logs/results/{time_now}_{MODEL}/"
    log_dir = f"logs/fit/{time_now}_{MODEL}/"

    os.makedirs(log_dir)
    os.makedirs(result_dir)

    # Dump used settings
    with open(f"{result_dir}settings.json", "x") as f:
        f.write(fit_settings.to_json(indent=4))
    with open(f"{result_dir}model_config.json", "x") as f:
        f.write(model_config.to_json(indent=4))
    for fold, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):
        xt = x_train[train_index]
        yt = pd.get_dummies(y_train[train_index])
        xv = x_train[val_index]
        yv = pd.get_dummies(y_train[val_index])

        # model = None
        # if MODEL == "CNN_PSP":
        #     model = CNN_PSP(xt[0].shape, PAD, STRIDE, POOL)
        #     # keras.utils.plot_model(model, "psp.png", show_shapes=True, expand_nested=True)
        # elif MODEL == "CNN_5":
        #     model = CNN_5(xt[0].shape, INCREASING)
        model = CNN(model_config)
        # model(xt[0:2])
        # x = keras.Input(xt[0].shape)
        # graph = keras.Model(inputs=x, outputs=model.call(x))
        # graph.summary()
        # keras.utils.plot_model(graph, "cnn.png", show_shapes=True, expand_nested=True)

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        saved_model = f"{result_dir}/checkpoint_{fold}"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(saved_model, save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=80),
            tf.keras.callbacks.TensorBoard(log_dir=f"{log_dir}/fold_{fold}", histogram_freq=1),
        ]

        # TODO: class weights IF option specified..
        history = model.fit(
            xt,
            yt,
            validation_data=(xv, yv),
            batch_size=fit_settings.batch_size,
            epochs=fit_settings.epochs,
            callbacks=callbacks,
            class_weight=class_weights,
        )

        # Dump training history
        with open(result_dir + f"/train_history_dict_{fold}", "wb") as f:
            pickle.dump(history.history, f)

        # TODO: Save test acc etc, and with better names so easier to compare
        model.load_weights(saved_model)
        prediction = np.argmax(model.predict(x_test), axis=1)
        visualize.visualize_test_result(y_test, prediction, result_dir + f"confusion_{fold}.svg")
        visualize.visualize_history(history.history, result_dir + f"history_{fold}.svg")
        return
