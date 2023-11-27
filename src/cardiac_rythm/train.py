import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from cardiac_rythm import visualize, preprocessing
from cardiac_rythm.models import CNN, CNNConfig
from dataclasses_json import dataclass_json
from sklearn import model_selection


@dataclass_json
@dataclass(frozen=True)
class FitSettings:
    epochs: int = 100
    batch_size: int = 32
    folds: int = 10
    normalize_data_length: bool = False  # Normalize, or weigh
    cross_validate: bool = False  # If true run all folds


def fit(file_path: str, fit_settings: FitSettings, model_config: CNNConfig) -> None:
    """Loads the data, model and fits it."""
    # Set random seed for reproducibility
    rng = np.random.RandomState(0)
    tf.random.set_seed(0)
    # Load data and create dataframe
    df = preprocessing.load_data(file_path)

    x = np.stack(df["s_ecg"].to_numpy())
    x = x.reshape((*x.shape, 1))
    y = df["c_label"].to_numpy()
    y = y - 1  # 0-4 instead of 1-5

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=rng,
        stratify=y,
    )

    # x_train, y_train = preprocessing.replicate_data(x_train, y_train)

    skf = model_selection.StratifiedKFold(n_splits=fit_settings.folds, random_state=rng, shuffle=True)

    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    result_dir = f"logs/results/{time_now}/"
    log_dir = f"logs/fit/{time_now}/"

    logging.info(f"Saving model to: {result_dir}")

    os.makedirs(log_dir)
    os.makedirs(result_dir)

    # Dump used settings
    with open(f"{result_dir}settings.json", "x") as f:
        f.write(fit_settings.to_json(indent=4))  # type: ignore
    with open(f"{result_dir}model_config.json", "x") as f:
        f.write(model_config.to_json(indent=4))  # type: ignore

    for fold, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):
        xt = x_train[train_index]
        yt = y_train[train_index]

        # TODO: Remove?
        # xt, yt = preprocessing.replicate_data(xt, yt)

        yt = pd.get_dummies(yt)
        xv = x_train[val_index]
        yv = pd.get_dummies(y_train[val_index])

        opt = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            # weight_decay=0.001 / fit_settings.epochs,
        )
        model = CNN(model_config)
        model.compile(
            opt,
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
            # class_weight=class_weights,
        )

        # Dump training history
        with open(result_dir + f"/train_history_dict_{fold}", "wb") as f:
            pickle.dump(history.history, f)

        # TODO: Save test acc etc, and with better names so easier to compare
        model.load_weights(saved_model)
        prediction = np.argmax(model.predict(x_test), axis=1)
        visualize.visualize_test_result(y_test, prediction, result_dir + f"confusion_{fold}.svg")
        visualize.visualize_history(history.history, result_dir + f"history_{fold}.svg")
        if not fit_settings.cross_validate:
            # Early stopping for testing.
            break
