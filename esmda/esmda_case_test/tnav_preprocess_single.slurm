#!/bin/bash
#SBATCH --job-name=tnav_pre_single
#SBATCH --output=logs/tnav_pre_single_%j.out
#SBATCH --error=logs/tnav_pre_single_%j.err
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

# ─── Environment ──────────────────────────────────────────────────────────────
module purge
module load python/3.9

# ─── Run ──────────────────────────────────────────────────────────────────────
mkdir -p logs

echo "Running tNav single preprocess (Job: $SLURM_JOB_ID)..."
python3 tnav_preprocess_single.py