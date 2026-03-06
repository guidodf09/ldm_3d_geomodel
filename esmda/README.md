# 🔁 ESMDA Workflow Automation

> Ensemble Smoother with Multiple Data Assimilation — automated, dependency-chained HPC job submission.

---

## Overview

This repository implements a fully automated **ES-MDA (Ensemble Smoother with Multiple Data Assimilation)** workflow for reservoir simulation. Jobs are submitted with dependency chaining so each stage automatically triggers the next upon successful completion.

The main entry point is **`submit_all_loop.sh`**, which orchestrates the entire pipeline from prior generation through iterative data assimilation.

---

## Quick Start

```bash
# Submit the full workflow
bash submit_all_loop.sh
```

That's it. All downstream jobs are scheduled automatically via job dependencies.

---

## Pipeline Architecture

The workflow runs as a linear chain of dependent jobs, with iterative ESMDA loops nested inside:

```
generate_priors                      # Generate prior ensemble models
└── esmda_true                       # Simulate the true/reference model
    └── tnav_preprocess_0            # Create run folders for iteration 0
        └── ensemble_0               # Run forward simulations (iteration 0)
            └── tnav_postprocess_0   # Extract simulation results
                └── iteration_0      # Update ESMDA variables
                    ├── tnav_preprocess_1
                    │   └── ensemble_1
                    │       └── tnav_postprocess_1
                    │           └── iteration_1
                    │               ├── tnav_preprocess_2
                    │               │   └── ...
                    │               │       └── iteration_N
                    │               └── ...
                    └── ...
```

Each ESMDA iteration `i` follows the same 4-step sub-loop:

```
tnav_preprocess_i  →  ensemble_i  →  tnav_postprocess_i  →  iteration_i
```

---

## Stage Descriptions

| Stage | Script / Job | Description |
|---|---|---|
| `generate_priors` | `generate_priors.sh` | Generates the initial ensemble of prior geological models |
| `esmda_true` | `esmda_true.sh` | Runs the reference/true model simulation for synthetic history matching |
| `tnav_preprocess_i` | `tnav_preprocess.sh` | Sets up tNavigator run folders for iteration `i` |
| `ensemble_i` | `submit_ensemble.sh` | Submits all ensemble member simulations in parallel |
| `tnav_postprocess_i` | `tnav_postprocess.sh` | Extracts and collects simulation outputs |
| `iteration_i` | `esmda_iteration.sh` | Performs the ES-MDA update step; prepares next iteration |

---

## Configuration

Key parameters are set at the top of `submit_all_loop.sh`:

```bash
N_ITERATIONS=4        # Number of ES-MDA iterations
ENSEMBLE_SIZE=100     # Number of ensemble members
QUEUE="gpu"           # HPC queue/partition to submit to
WORKDIR="/path/to/workdir"
```

---

## Output Structure

```
workdir/
├── priors/                  # Initial ensemble models
├── true_model/              # Reference simulation results
├── iteration_0/
│   ├── run_folders/         # tNavigator input decks
│   ├── results/             # Raw simulation outputs
│   └── updated_models/      # Post-update ensemble
├── iteration_1/
│   └── ...
└── iteration_N/
    └── final_models/        # Final assimilated ensemble
```

---

## Dependencies

- **tNavigator** — reservoir simulator
- **Python ≥ 3.8** — for ES-MDA update scripts
- **NumPy / SciPy** — ensemble math
- A job scheduler: **SLURM** or **PBS/Torque**

---

## Notes

- If any job in the chain fails, downstream jobs will not run (dependency not satisfied). Check job logs in `workdir/logs/`.
- To re-run from a specific iteration, comment out earlier stages in `submit_all_loop.sh` and set the starting iteration index manually.
- The number of `alpha` coefficients in ES-MDA must match `N_ITERATIONS` — see `esmda_iteration.sh` for configuration.

---

## License

MIT — see `LICENSE` for details.
