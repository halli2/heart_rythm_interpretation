"""This file contains classes and methods to search for hyperparameters in the model."""

from __future__ import annotations

import copy
import logging
import pickle
import pprint as pp
from argparse import ArgumentParser

import keras
import numpy as np
import pandas as pd
from keras_tuner import HyperModel, HyperParameters, RandomSearch
from keras_tuner.src.engine import tuner_utils
from sklearn.model_selection import StratifiedKFold, train_test_split

from cardiac_rythm.models import CNN, CNNConfig
from cardiac_rythm.preprocessing import load_data
from cardiac_rythm.visualize import visualize_history, visualize_test_result


class CNNTuner(HyperModel):
    """This class configures the tuneable hyperparameters we care about.

    Args:
        n_filters: int, the number of filters we want to test
        **kwargs: Arguments to inherited `HyperModel`
    """

    def __init__(self, n_filters: int, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.filter_choice = [5, 10, 15, 20]
        self.kernel_choice = [5, 10, 15, 20]

    def build(self, hp: HyperParameters) -> CNN:
        filters = []
        kernels = []
        strides = []
        pool = []
        for i in range(self.n_filters):
            filters.append(hp.Choice(f"filters{i}", self.filter_choice))
            kernels.append(hp.Choice(f"kernels{i}", self.kernel_choice))
            strides.append(1)
            pool.append([2, 2])

        model_config = CNNConfig(
            filters,
            kernels,
            strides,
            pool,
            "valid",
            fc_end=[64, 128],
            dropout=hp.Float("dropout", 0.1, 0.9, step=0.1),
        )

        model = CNN(model_config)
        opt = keras.optimizers.Adam(
            learning_rate=0.001,
            # weight_decay=0.001 / fit_settings.epochs,
        )
        model.compile(
            opt,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class RandomSearchOptimization(RandomSearch):
    """Override RandomSearch so we can run cross validation + saving some extra stuff
    like plots and history dicts

    Args:
        rng: RandomState, numpy RandomState
        n_folds: int, number of folds in the cross validation
        **kwargs: arguments relevant to `RandomSearch` and relevant `Tuners` under `RandomSearch`
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        n_folds: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.skf = StratifiedKFold(n_splits=n_folds, random_state=rng, shuffle=True)

    def run_trial(self, trial, *args, **kwargs):
        """Override `run_trial` to implement cross validation."""
        x_train, y_train, x_test, y_test, *remaining_args = args

        trial_dir = self.get_trial_dir(trial_id=trial.trial_id)
        # Not using `ModelCheckpoint` to support MultiObjective.
        # It can only track one of the metrics to save the best model.
        saved_model = self._get_checkpoint_fname(trial.trial_id)

        # TODO: Save the model that perfomrs best on the test set?
        model_checkpoint = tuner_utils.SaveBestEpoch(
            objective=self.oracle.objective,
            filepath=saved_model,
        )
        original_callbacks = kwargs.pop("callbacks", [])

        histories = []
        for execution in range(self.executions_per_trial):
            total_acc = 0.0
            total_val_acc = 0.0
            total_loss = 0.0
            total_val_loss = 0.0
            for fold, (train_index, val_index) in enumerate(self.skf.split(x_train, y_train)):
                xt, xv = x_train[train_index], x_train[val_index]
                yt, yv = y_train[train_index], y_train[val_index]
                yt = pd.get_dummies(yt)
                yv = pd.get_dummies(yv)
                copied_kwargs = copy.copy(kwargs)
                callbacks = self._deepcopy_callbacks(original_callbacks)

                # Add fold to tensorboard callback
                for callback in callbacks:
                    if isinstance(callback, keras.callbacks.TensorBoard):
                        callback.log_dir += f"/fold_{fold}"

                self._configure_tensorboard_dir(callbacks, trial, execution)
                callbacks.append(tuner_utils.TunerCallback(self, trial))
                # Only checkpoint the best epoch across all executions.
                callbacks.append(model_checkpoint)
                copied_kwargs["callbacks"] = callbacks

                hp = trial.hyperparameters
                model: CNN = self._try_build(hp)

                # Dump the model config in a better format
                if fold == 0:
                    logging.info(f"Training: {pp.pformat(model.config)}")
                    with open(f"{trial_dir}/model_config.json", "x") as f:
                        f.write(model.config.to_json(indent=2))
                history = self.hypermodel.fit(
                    hp,
                    model,
                    xt,
                    yt,
                    validation_data=(xv, yv),
                    *remaining_args,  # noqa: B026
                    **copied_kwargs,
                )

                # Test the model and save conf mat, accuracy and loss
                # TODO: Or just do it on the BEST model?
                visualize_history(history.history, f"{trial_dir}/{fold}.svg")
                # Dump history in case we want to plot it.
                with open(f"{trial_dir}/history_dict_{fold}", "wb") as f:
                    pickle.dump(history.history, f)

                total_loss += min(history.history["loss"])
                total_val_loss += min(history.history["val_loss"])
                total_acc += max(history.history["accuracy"])
                total_val_acc += max(history.history["val_accuracy"])

            # Store average loss/accuracy
            avg_loss = total_loss / self.skf.get_n_splits()
            avg_val_loss = total_val_loss / self.skf.get_n_splits()
            avg_accuracy = total_acc / self.skf.get_n_splits()
            avg_val_accuracy = total_val_acc / self.skf.get_n_splits()

            histories.append(
                {
                    "loss": avg_loss,
                    "val_loss": avg_val_loss,
                    "accuracy": avg_accuracy,
                    "val_accuracy": avg_val_accuracy,
                }
            )
            model.load_weights(saved_model)
            prediction = np.argmax(model.predict(x_test), axis=1)
            visualize_test_result(y_test, prediction, f"{trial_dir}/conf_mat.svg")

        return histories


def search_for_hyperparameters(file_path: str, rng: np.random.RandomState) -> None:
    df = load_data(file_path)

    x = np.stack(df["s_ecg"].to_numpy())
    x = x.reshape((*x.shape, 1))
    y = df["c_label"].to_numpy()
    y = y - 1  # 0-4 instead of 1-5

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=rng,
        stratify=y,
    )
    n_filters = 2
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=50,
        ),
        keras.callbacks.TensorBoard(
            log_dir=f"results/{n_filters}/tb",
        ),
    ]

    tuner = RandomSearchOptimization(
        rng,
        n_folds=2,  # TODO: 10 10 10
        hypermodel=CNNTuner(n_filters=n_filters),
        objective="val_loss",
        max_trials=2,
        directory="results",
        project_name=f"{n_filters=}",
        overwrite=True,  # TODO: FALSE FALSE FALSE! (don't want to lose data after many trials..)
    )
    tuner.search(
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=32,
        epochs=50,
        callbacks=callbacks,
    )
    # TODO: Check the best results etc..
    with open(f"results/{n_filters=}/tuner.pickle", "wb") as f:
        pickle.dump(tuner, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = ArgumentParser(description="Search for hyperparameters.")
    parser.add_argument("file_path", help="Path to .mat file.")
    args = parser.parse_args()

    import tensorflow as tf

    rng = np.random.RandomState(0)
    tf.random.set_seed(0)

    search_for_hyperparameters(args.file_path, rng)
