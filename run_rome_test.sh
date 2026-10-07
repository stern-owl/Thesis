#!/bin/bash

#SBATCH --job-name=rome_single
#SBATCH --output=/localstorage/home/f20220930/rome_single_%j.out
#SBATCH --error=/localstorage/home/f20220930/rome_single_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=h100-mig
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1

echo "===== JOB INFO ====="
date
hostname

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== ENVIRONMENT ====="
source /localstorage/home/f20220930/miniforge3/etc/profile.d/conda.sh
conda activate EasyEdit

which python
python --version

echo
echo "===== RUNNING ROME ====="
cd /localstorage/home/f20220930/EasyEdit

export PYTHONPATH="$PWD:$PYTHONPATH"
export MPLCONFIGDIR=/localstorage/home/f20220930/.config/matplotlib
mkdir -p "$MPLCONFIGDIR"
python experiments/rome_single_test.py

echo
echo "===== JOB FINISHED ====="
date
