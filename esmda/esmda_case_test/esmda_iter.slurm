#!/bin/bash
#SBATCH --job-name=esmda_iter
#SBATCH --output=logs/esmda_iter_%j.out
#SBATCH --error=logs/esmda_iter_%j.err
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# ─── Environment ──────────────────────────────────────────────────────────────
module purge

module load python/3.9
module load cuda/11.5
module load cudnn/8.3.3.40
module load openmpi/4.1.2
module load libfabric/1.14
module load ucx/1.12.1
module load gcc/10.1.0
module load py-mpi4py/3.1.3_py39
module load openblas/0.3.4
module load zlib/1.2.11
module load szip/2.1.1
module load hdf5/1.12.0

# ─── Validate ─────────────────────────────────────────────────────────────────
ENSEMBLE_ID="${1:?ERROR: ENSEMBLE_ID not provided}"

mkdir -p logs

# ─── Run ──────────────────────────────────────────────────────────────────────
echo "Running ES-MDA iteration for ensemble member $ENSEMBLE_ID (Job: $SLURM_JOB_ID)..."

python3 esmda_iter.py --index "$ENSEMBLE_ID"