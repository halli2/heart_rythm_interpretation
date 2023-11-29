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
        n_filters: the number of filters we want to test
        filter_choice: possible choice of filters to use in the search
        kernel_choice: possible choice of kernel sizes to use in the search
        dropout_choice: tuple of lower dropout choice and highest dropout choice (step=0.1)
        pool_choice: possible choices for pool sizes
        stride: stride used in convolution
        n_fc: number of fully connected hidden layers
        fc_choice: possible choices of nodes in the fully connected hidden layers
        **kwargs: Arguments to inherited `HyperModel`
    """

    def __init__(
        self,
        n_filters: int,
        filter_choice: list[int],
        kernel_choice: list[int],
        dropout_choice: tuple[float, float],
        pool_choice: list[int],
        stride: int,
        n_fc: int,
        fc_choice: list[int],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.filter_choice = filter_choice
        self.kernel_choice = kernel_choice
        self.dropout_choice = dropout_choice
        self.pool_choice = pool_choice
        self.stride = stride
        self.n_fc = n_fc
        self.fc_choice = fc_choice

    def build(self, hp: HyperParameters) -> CNN:
        filters = []
        kernels = []
        strides = []
        pool = []
        pool_choice = hp.Choice("pool", self.pool_choice)
        for i in range(self.n_filters):
            filters.append(hp.Choice(f"filters{i}", self.filter_choice))
            kernels.append(hp.Choice(f"kernels{i}", self.kernel_choice))
            strides.append(self.stride)
            pool.append(pool_choice)

        fc_blocks = []
        for i in range(self.n_fc):
            fc_blocks.append(hp.Choice(f"fc{i}", self.fc_choice))
        model_config = CNNConfig(
            filters,
            kernels,
            strides,
            pool,
            "valid",
            fc_end=fc_blocks,
            dropout=hp.Float(
                "dropout",
                self.dropout_choice[0],
                self.dropout_choice[1],
                step=0.1,
            ),
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


class RandomSearchCrossValidate(RandomSearch):
    """Override RandomSearch so we can run cross validation + saving some extra stuff
    like plots and history dicts

    Args:
        rng: numpy RandomState used in `StratifiedKFold`
        n_folds: number of folds in the cross validation
        **kwargs: arguments relevant to `RandomSearch` and relevant `Tuners` under `RandomSearch`
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        n_folds: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.skf = StratifiedKFold(n_splits=n_folds, random_state=rng, shuffle=True)

    def run_trial(self, trial, *args, **kwargs) -> list[dict]:
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
        # Always 1 execution
        total_acc = 0.0
        total_val_acc = 0.0
        total_loss = 0.0
        total_val_loss = 0.0
        for fold, (train_index, val_index) in enumerate(
            self.skf.split(x_train, y_train)
        ):
            xt, xv = x_train[train_index], x_train[val_index]
            yt, yv = y_train[train_index], y_train[val_index]
            yt = pd.get_dummies(yt)
            yv = pd.get_dummies(yv)
            copied_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)

            # Set TB dir to use current fold as execution_nn
            self._configure_tensorboard_dir(callbacks, trial, fold)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            callbacks.append(model_checkpoint)
            copied_kwargs["callbacks"] = callbacks

            hp = trial.hyperparameters
            model: CNN = self._try_build(hp)

            # Dump the model config in a better format (if it is valid)
            if fold == 0:
                logging.info(f"Training: {pp.pformat(model.config)}")
                with open(f"{trial_dir}/model_config.json", "x") as f:
                    f.write(model.config.to_json(indent=2))  # type: ignore
            try:
                history = self.hypermodel.fit(
                    hp,
                    model,
                    xt,
                    yt,
                    validation_data=(xv, yv),
                    *remaining_args,  # noqa: B026
                    **copied_kwargs,
                )
            except ValueError as e:
                logging.warning(f"Model: {model.config} had error calling 'fit'")
                logging.warning(e)
                histories.append(
                    {
                        "loss": 9999,
                        "val_loss": 9999,
                        "accuracy": 0,
                        "val_accuracy": 0,
                    }
                )
                return histories

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
        # TODO: Do this on every fold + save model for each?
        model.load_weights(saved_model)
        prediction = np.argmax(model.predict(x_test), axis=1)
        visualize_test_result(y_test, prediction, f"{trial_dir}/conf_mat.svg")

        return histories


# As running a lot of tests will produce a lot of output (hundreds of megabytes)
# We create a callback to only get some outputs
class ProgressionLogger(keras.callbacks.Callback):
    """
    Keras callback to log on the n'th epoch. Useful when logging multiple runs to a file, as the default `ProgbarLogger`
    will spit out enormous amount of info

    args:
        show: show stats on n'th epoch
    """

    def __init__(self, show: int = 10):
        super().__init__()
        self.counter = 0
        self.show = show

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if self.counter == self.show or epoch == 1:
            if epoch > 1:
                self.counter = 0
            logging.info(
                f"Epoch: {epoch} - Loss: {logs['loss']} - ValLoss: {logs['val_loss']} - "
                f"Accuracy: {logs['accuracy']} - ValAccuracy: {logs['val_accuracy']}"
            )
        self.counter += 1


def search_for_hyperparameters(args, rng: np.random.RandomState) -> None:
    df = load_data(args.file_path)

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
    n_filters = args.n_filters
    n_fc = args.n_fc
    project_name = f"{n_filters=}_{n_fc=}"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=30,
        ),
        keras.callbacks.TensorBoard(
            log_dir=f"results/{project_name}/tb",
        ),
        ProgressionLogger(),
    ]

    hp = HyperParameters()
    if args.override:
        hp.Fixed("fc0", value=64)
        hp.Fixed("fc1", value=32)
        hp.Fixed("dropout", value=0.3)
        project_name = f"{project_name}_overriden"
    tuner = RandomSearchCrossValidate(
        rng,
        n_folds=args.n_folds,
        hyperparameters=hp,
        hypermodel=CNNTuner(
            args.n_filters,
            args.filters,
            args.kernels,
            args.dropout,
            args.pool,
            args.stride,
            args.n_fc,
            args.fc_choice,
        ),
        objective="val_loss",
        max_trials=args.max_trials,
        directory="results",
        project_name=project_name,
        overwrite=False,
    )
    tuner.search(
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=32,
        epochs=250,
        callbacks=callbacks,
        verbose=0,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = ArgumentParser(description="Search for hyperparameters.")
    parser.add_argument("file_path", help="Path to .mat file.")
    # Tuner settings
    parser.add_argument("--max_trials", type=int, default=5)
    parser.add_argument("--n_folds", type=int, default=10)

    # Model settings
    parser.add_argument("--n_filters", type=int, default=2)
    parser.add_argument("--filters", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--kernels", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.1, 0.9])
    parser.add_argument("--pool", type=int, nargs="+", default=[2])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n_fc", type=int, default=2)
    parser.add_argument("--fc_choice", type=int, nargs="+", default=[32, 64, 128])

    parser.add_argument(
        "--override", help="Override FC and Dropout", action="store_true"
    )
    args = parser.parse_args()


    rng = np.random.RandomState(0)
    # tf.random.set_seed(0)

    search_for_hyperparameters(args, rng)
