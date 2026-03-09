#!/bin/bash
# ─── Setup ────────────────────────────────────────────────────────────────────
mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"

read_cfg() { python3 -c "import yaml; cfg=yaml.safe_load(open('${CONFIG}')); print($1)"; }

export RUN_PATH=$(read_cfg        "cfg['paths']['run_path']")
export SINGLE_RUN_PATH=$(read_cfg "cfg['paths']['single_run_path']")
Na=$(read_cfg                     "cfg['ensemble']['Na']")
NR=$(read_cfg                     "cfg['ensemble']['nr']")

SKIP_TRUE=${1:-0}  # Pass 1 as first argument to skip true model: ./submit.sh 1


# ─── Helper ───────────────────────────────────────────────────────────────────
submit() {
    local name="$1"; shift
    local jobid=$(sbatch "$@" | awk '{print $4}')
    echo "${name} submitted with Job ID: ${jobid}" >&2
    echo "$jobid"
}

dep() { echo "--dependency=afterok:$1"; }

# ─── Initial Chain ────────────────────────────────────────────────────────────
priors_jobid=$(submit "Generate priors" generate_priors.slurm)

if [[ "$SKIP_TRUE" -eq 1 ]]; then
    echo "Skipping true model simulation — using existing data in ${SINGLE_RUN_PATH}"
    first_dep_jobid="$priors_jobid"
else
    true_jobid=$(submit "True model" --export=SINGLE_RUN_PATH $(dep $priors_jobid) esmda_true.slurm)
    first_dep_jobid="$true_jobid"
fi

# ─── Iteration Function ───────────────────────────────────────────────────────
run_iteration() {
    local i="$1"
    local prev_jobid="$2"

    tnav_preprocess_jobid=$(submit  "Tnav preprocess $i"  --export=RUN_PATH $(dep $prev_jobid)                          tnav_preprocess.slurm "$i")
    ensemble_jobid=$(submit         "Ensemble $i"         --export=RUN_PATH --array=0-$((NR-1))%$NR $(dep $tnav_preprocess_jobid) esmda_ensemble.slurm "$i")
    tnav_postprocess_jobid=$(submit "Tnav postprocess $i" --export=RUN_PATH $(dep $ensemble_jobid)                      tnav_postprocess.slurm "$i")
    iter_jobid=$(submit             "Iteration $i"                          $(dep $tnav_postprocess_jobid)              esmda_iter.slurm "$i")

    echo "$iter_jobid"
}

# ─── Run All Iterations ───────────────────────────────────────────────────────
iter_jobid=$(run_iteration 0 "$first_dep_jobid")

for i in $(seq 1 $Na); do
    iter_jobid=$(run_iteration "$i" "$iter_jobid")
done