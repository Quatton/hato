#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o result.log
#SBATCH -e result.err
#SBATCH --gpus 1

source .venv/bin/activate
# python -m pip install -e . --no-build-isolation

# Update this path to where you extract the DTU dataset
python src/training/train.py
# python render.py -m output/2025-04-19/scan105 -r 2 --depth_ratio 1 --skip_test --skip_train