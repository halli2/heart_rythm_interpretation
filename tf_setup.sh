#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=02:00:00
#SBATCH --job-name=tf_setup
#SBATCH --output=tf_setup.out

# Set up environment.
uenv verbose cuda-11.8.0 cudnn-11.x-8.6.0
uenv verbose TensorRT-11.x-8.6-8.5.3.1
uenv verbose miniconda3-py310
conda create --name tf_env python=3.10
conda activate tf_env
# Cuda
pip install nvidia-cudnn-cu11==8.6.0.163 tensorrt==8.5.3.1 tensorflow==2.12.0 
# Project deps
pip install .
echo "Finished setting up."
