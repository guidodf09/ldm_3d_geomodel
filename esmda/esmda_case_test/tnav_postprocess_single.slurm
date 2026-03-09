#!/bin/bash
#SBATCH --job-name=tnav_post_single
#SBATCH --output=logs/tnav_post_single_%j.out
#SBATCH --error=logs/tnav_post_single_%j.err
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

# ─── Environment ──────────────────────────────────────────────────────────────
module purge
module load python/3.9

# ─── Run ──────────────────────────────────────────────────────────────────────
mkdir -p logs

echo "Running tNav single postprocess (Job: $SLURM_JOB_ID)..."
python3 tnav_postprocess_single.py

# ─── Cleanup ──────────────────────────────────────────────────────────────────
echo "Cleaning up logs..."
rm -f logs/*.log
rm -f logs/ens*