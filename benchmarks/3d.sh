#!/bin/bash
#SBATCH --job-name=watershed_bench
#SBATCH --output=logs/moaap_benchmark_%j.out
#SBATCH --error=logs/moaap_benchmark_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=64
#SBATCH --constraint=EPYC_7742
#SBATCH --mem-per-cpu=6500MB

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate your conda/virtual environment
source ~/sem_venv/bin/activate

python run_3d.py
