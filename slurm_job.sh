#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=02:00:00
#SBATCH --job-name=cardiac_fit
#SBATCH --output=slurm_job.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.6.0
uenv verbose TensorRT-11.x-8.6-8.5.3.1
uenv verbose miniconda3-py310
conda activate tf_env

CONDA_PREFIX=$VIRTUAL_ENV
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
#pip install .

# Run the Python script that uses the GPU
python -u src/cardiac_rythm  "/home/prosjekt/BMDLab/matlab/resprog/GUI/CleanCutsDL/cutDataCinCTTI_rev_v2.mat"
echo "--Finished--"