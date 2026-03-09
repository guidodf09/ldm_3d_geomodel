#!/bin/bash
#SBATCH --job-name=esmda_true
#SBATCH --output=logs/true_%j.out
#SBATCH --error=logs/true_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=serc
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16

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
export SINGLE_RUN_PATH=$(python3 -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(cfg['paths']['single_run_path'])")

# ─── Validate ─────────────────────────────────────────────────────────────────
mkdir -p logs

INPUT_FILE="${SINGLE_RUN_PATH}/run_0/test.DATA"

# ─── Preprocess ───────────────────────────────────────────────────────────────
echo "Running true model preprocess (Job: $SLURM_JOB_ID)..."
python3 tnav_preprocess_single.py

# ─── tNavigator ───────────────────────────────────────────────────────────────
module use /home/groups/sh_s-dss/share/sdss/modules/modulefiles
module load tnavigator/24.3
module load hdf5/1.12.2  

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE" >&2
    exit 1
fi

echo "Running tNavigator..."
tNav-con \
    -n "$OMP_NUM_THREADS" \
    --ecl-rsm \
    --license-settings="${TNAVIGATOR_LICENSE_PATH}" \
    "$INPUT_FILE" \
    > "logs/out_true.log"

# ─── Postprocess ──────────────────────────────────────────────────────────────
echo "Running true model postprocess..."
python3 tnav_postprocess_single.py