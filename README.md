# Emergency medical data analysis - heart-rythm-interpretation

Continuation of [https://github.com/SanderSondeland/ELE690_project1](https://github.com/SanderSondeland/ELE690_project1)

## Filestructure

* notebooks - contains notebooks for vizualisation of data.
* cardiac_rythm - contains the code for the models.
* logs - containing logs and results form the model fitting.

## How to run

### With hatch:

```sh
hatch run fit
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
sbatch slurm_job.sh
```

## Results

Results are stored in the logs folder. Confusion matrix, and history plots are stored in logs/results/{timestamp}_{modelname}.
To vizualise more data with tensorboard run `tensorboard --logdir logs/fit/{timestamp}_{modelname}` while in venv or poetry shell.
