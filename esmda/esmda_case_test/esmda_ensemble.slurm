#!/bin/bash
#SBATCH --job-name=esmda_ens
#SBATCH --output=logs/ens_%A_%a.out
#SBATCH --error=logs/ens_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --partition=serc
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# ─── Array size from config ───────────────────────────────────────────────────
# SBATCH --array is set dynamically below via --array flag in submit.sh
# nr is read here for validation only
NR=$(python3 -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(cfg['ensemble']['nr'])")
#SBATCH --array=0-${NR}%${NR}

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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ─── Validate Inputs ──────────────────────────────────────────────────────────
ENSEMBLE_ID="${1:?ERROR: ENSEMBLE_ID not provided}"

if [[ -z "$RUN_PATH" ]]; then
    echo "ERROR: RUN_PATH is not set" >&2
    exit 1
fi

# ─── tNavigator ───────────────────────────────────────────────────────────────
module use /home/groups/sh_s-dss/share/sdss/modules/modulefiles
module load tnavigator/24.3
module load hdf5/1.12.2   

mkdir -p logs

INPUT_FILE="${RUN_PATH}/run_${SLURM_ARRAY_TASK_ID}/test.DATA"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE" >&2
    exit 1
fi

echo "Running ensemble member $ENSEMBLE_ID (array task $SLURM_ARRAY_TASK_ID)..."

tNav-con \
    -n "$OMP_NUM_THREADS" \
    --ecl-rsm \
    --license-settings="${TNAVIGATOR_LICENSE_PATH}" \
    "$INPUT_FILE" \
    > "logs/out_${SLURM_ARRAY_TASK_ID}.log"