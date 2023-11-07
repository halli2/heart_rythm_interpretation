#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=24:00:00
#SBATCH --job-name=cardiac_hyper_fit
#SBATCH --output=hyper_fit.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.6.0
uenv verbose TensorRT-11.x-8.6-8.5.3.1
uenv verbose miniconda3-py310
conda activate tf_env

#pip install .

python -u src/cardiac_rythm "/home/prosjekt/BMDLab/matlab/resprog/GUI/CleanCutsDL/cutDataCinCTTI_rev_v2.mat" \
--filters 5 25 40 40 50 \
--kernels 40 40 20 10 5 \
--stride 1 \
--padding "valid" \
--pool 2 \
--fc_end 64 32 \
--epochs 250 \
--dropout 0.3 \
--batch_size 32 \
--cross_validate

echo "--Finished--"
