import os
from datetime import datetime

import numpy as np
import pandas as pd
import preprocessing
import tensorflow as tf
import visualize
from models import CNN_5, CNN_PSP
from sklearn import model_selection

NORMALIZE_DATA_LENGTH = False
CLASS_WEIGHTS = True
EPOCHS = 100
BATCH_SIZE = 32
FOLDS = 10
MODEL = "CNN_5"  # CNN_PSP
PAD = True
STRIDE = False
POOL = 1


def fit() -> None:
    """Loads the data, model and fits it."""
    # Set random seed for reproducibility
    rng = np.random.RandomState(0)
    # Load data and create dataframe
    df = preprocessing.load_data()

    if NORMALIZE_DATA_LENGTH:
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

    skf = model_selection.StratifiedKFold(n_splits=FOLDS, random_state=rng, shuffle=True)

    result_dir = f"logs/results/{datetime.now().strftime('%Y%m%d-%H%M%S')}_{MODEL}/"
    log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}_{MODEL}/"
    os.makedirs(log_dir)
    os.makedirs(result_dir)
    for fold, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):
        xt = x_train[train_index]
        yt = pd.get_dummies(y_train[train_index])
        xv = x_train[val_index]
        yv = pd.get_dummies(y_train[val_index])

        model = None
        if MODEL == "CNN_PSP":
            model = CNN_PSP(xt[0].shape, PAD, STRIDE, POOL)
        elif MODEL == "CNN_5":
            model = CNN_5(xt[0].shape, True)

        model.summary()
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        saved_model = f"{result_dir}/checkpoint_{fold}.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(saved_model, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=80),
            tf.keras.callbacks.TensorBoard(log_dir=f"{log_dir}/fold_{fold}", histogram_freq=1),
        ]

        # TODO: class weights IF option specified..
        history = model.fit(
            xt,
            yt,
            validation_data=(xv, yv),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,
        )

        model.load_weights(saved_model)
        prediction = np.argmax(model.predict(x_test), axis=1)
        visualize.visualize_test_result(y_test, prediction, result_dir + f"confusion_{fold}.svg")
        visualize.visualize_history(history.history, result_dir + f"history_{fold}.svg")
