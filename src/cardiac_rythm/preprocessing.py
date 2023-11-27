import logging

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from sklearn import model_selection


def load_data(data_file: str) -> pd.DataFrame:
    """Load the data. Data_file is a .mat file (cutDataCinCTTI_rev_v2.mat)"""
    mat_cut_data: NDArray = scipy.io.loadmat(
        data_file,
        simplify_cells=True,
    )["data"]

    df = pd.DataFrame(mat_cut_data)
    return df


def load_train_test_data(data_file: str) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    df = load_data(data_file)
    x = np.stack(df["s_ecg"].to_numpy())
    x = x.reshape((*x.shape, 1))
    y = df["c_label"].to_numpy()
    y = y - 1  # 0-4 instead of 1-5

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=0,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test


def replicate_data(x, y):
    vt = 4
    asy = 0

    uniques, nunique = np.unique(y, return_counts=True)
    # TODO: Turn to debug?
    logging.info("Normalizing training data by duplicating.")
    logging.info(f"From: {uniques=} - {nunique=}")

    x_vt = x[np.where(y == vt)]
    y_vt = y[np.where(y == vt)]
    x_asy = x[np.where(y == asy)]
    y_asy = y[np.where(y == asy)]

    x = np.vstack((x, x_asy))
    y = np.hstack((y, y_asy))

    for _ in range(4):
        x = np.vstack((x, x_vt))
        y = np.hstack((y, y_vt))

    uniques, nunique = np.unique(y, return_counts=True)
    # TODO: Turn to debug?
    logging.info("Normalizing training data by duplicating.")
    logging.info(f"From: {uniques=} - {nunique=}")

    return x, y


# TODO: Er dette nødvendig?? Er dette bare tap av data? TODO: TEST!
def normalize_data_length(df: pd.DataFrame) -> pd.DataFrame:
    class_value_count = df["c_label"].value_counts().sort_index()
    classes = class_value_count.index
    print(f"Number of occurences of each consensus class: {class_value_count}")
    min_class_count = min(class_value_count.values)

    shortened = [
        df[df["c_label"] == class_val].sample(min_class_count) for class_val in classes
    ]
    df_shortened = pd.concat(shortened, ignore_index=True)
    return df_shortened
