#!/bin/bash
#SBATCH --job-name=tnav_post
#SBATCH --output=logs/tnav_post_%j.out
#SBATCH --error=logs/tnav_post_%j.err
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

# ─── Environment ──────────────────────────────────────────────────────────────
module purge
module load python/3.9

# ─── Validate ─────────────────────────────────────────────────────────────────
ENSEMBLE_ID="${1:?ERROR: ENSEMBLE_ID not provided}"

mkdir -p logs

# ─── Run ──────────────────────────────────────────────────────────────────────
echo "Running tNav postprocess for ensemble member $ENSEMBLE_ID (Job: $SLURM_JOB_ID)..."
python3 tnav_postprocess.py --index "$ENSEMBLE_ID"

# ─── Cleanup ──────────────────────────────────────────────────────────────────
echo "Cleaning up logs..."
rm -f logs/*.log
rm -f logs/ens*