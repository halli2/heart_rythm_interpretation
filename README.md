# Heart rhythm interpretation using a convolutional neural network

Continuation of [https://github.com/SanderSondeland/ELE690_project1](https://github.com/SanderSondeland/ELE690_project1)

## Filestructure

* src/notebooks - contains notebooks for vizualisation of data.
* src/cardiac_rythm - contains the code for the models.
* results - contains the results for the hyper parameter search
* logs - containing logs and results from the model fitting.
* report - contains the report from this project in pdf format, and latex code with figures.

The "best" model is added under `logs/results/best_model`, which contains checkpoints for every fold done under
cross validation and model settings in `model_config.json`.

## How to run

To set up the project locally either pdm or pip/venv can be used.
For usage on gorina a couple scripts are included.

### Setup

This project used [pdm](https://github.com/pdm-project/pdm) as a package
manager, which can be used to set up the project with this command:

```sh
pdm install
```

If not using pdm it can just as easily be setup with pip/venv.
These commands will setup a virtual environment and install all dependencies
as configured in `pyproject.toml`.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

### Local running

Example of how to run a specific model. We call the script with the path to dataset and
arguments to tell how the model should be configured:

```sh
python -u src/cardiac_rythm "/path/to/cutDataCinCTTI_rev_v2.mat" \
--filters 64 32 16 \
--kernels 40 20 10 \
--stride 1 \
--padding "valid" \
--pool 2 \
--fc_end 64 32 \
--epochs 250 \
--dropout 0.3 \
--batch_size 32
```

To get all available options run `python -u src/cardiac_rythm --help`.

Example of how to run the hyper parameter random seach, this can also be
called with `--help` to get all available commands:

```sh
python -u src/cardiac_rythm/hyper_tune.py "/path/to/cutDataCinCTTI_rev_v2.mat"  \
--max_trials 250 \
--n_folds 10 \
--n_filters 2 \
--filters 5 10 15 20 25 30 40 50 \
--kernels 5 10 15 20 25 30 40 50 \
--dropout 0.1 0.9 \
--pool 2 \
--stride 1 \
--n_fc 2 \
--fc_choice 16 32 64 128
```

Explanation of whats happening.
These commands will start the random search with 250 trials, 10-fold crossvalidation, 2 filters (2 convolutional blocks), number of filters in each block and kernels
will choose randomly from the given values, dropout will be between 0.1 and 0.9.
Pool size set to 2, stride set to 1 and 2 fully connected layers at the end choosing between the given
sizes.

### On Gorina (internal servers University of Stavanger)

The file `tf_setup.sh` contains commands to set up the project on gorina.
The files `slurm_man_fit.sh` and `slurm_job_hyper.sh` contains commands to 
setup the environment to run on the gpu and respectively manually train a model
and to run the hyperparameter random search. 

#### On gorina11 internal server using slurm job:

Setup:
```sh
sbatch tf_setup.sh
```

Run:
```sh
sbatch slurm_man_fit.sh
sbatch slurm_job_hyper.sh
```

## Notebooks

Notebook `data_vizualisation.ipynb` contains vizualisations of the dataset and structures,
`results_hyper.ipynb` contains vizualisations of the results from the random search, 
and `results.ipynb` contains code to extract information from one model.
