# Emergency medical data analysis - heart-rythm-interpretation

Continuation of [https://github.com/SanderSondeland/ELE690_project1](https://github.com/SanderSondeland/ELE690_project1)

## Filestructure

* src/notebooks - contains notebooks for vizualisation of data.
* src/cardiac_rythm - contains the code for the models.
* results - contains the results for the hyper parameter search
* logs - containing logs and results form the model fitting.

## How to run

### With pdm:

```sh
pdm install
pdm run main
```

### Using pip and venv:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install .
python src/cardiac_rythm
```

### On gorina11 internal server using slurm job:

Setup:
```sh
sbatch tf_setup.sh
```

Run:
```sh
sbatch slurm_man_fit.sh
sbatch slurm_job_hyper.sh
```

Where `slurm_man_fit.sh` contains an example of how to run a specific model and `slurm_job_hyper.sh`
contains an example of how to run random search given some parameters.
