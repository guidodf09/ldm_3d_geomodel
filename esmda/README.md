# ES-MDA Latent Diffusion Model — Workflow

> Ensemble Smoother with Multiple Data Assimilation — automated, dependency-chained HPC job submission.

---

## Problem Description
This workflow addresses probabilistic history matching for a 3D channelized oil-water reservoir. The reservoir is modeled on a 128×128×32 grid with three facies types (background, channel, high-permeability channel) controlling permeability and porosity distribution.

The field operates 6 producers and 3 injectors under BHP control. Producers target 4300 psi and water injectors 4700 psi. The simulation runs for 1500 days (30 × 50-day steps). Observed data consists of oil production rates (WOPR), water production rates (WWPR), and water injection rates (WWIR) at all wells.

The goal is to update an ensemble of geomodels such that their simulated production responses match the observed data from a reference (true) model, using ES-MDA as the assimilation algorithm and an LDM to ensure updated models remain geologically consistent.

Reservoir simulations are run with tNavigator using `test.DATA` as the input file, which reads facies-derived `PERMX.INC` and `PORO.INC` files written by the preprocessing step.

---

## Overview
The ES-MDA update is performed in the latent space of a pretrained LDM. At each iteration, the ensemble of latent vectors is updated via ES-MDA using the mismatch between simulated and observed production data, obtained from simulating the corresponding LDM-generated models. All parameters are centralized in `config.yaml`.

Jobs are submitted with dependency chaining so each stage automatically triggers the next upon successful completion. The main entry point is `submit_all_loop.sh`, which contains the entire pipeline from prior generation through iterative data assimilation. To save on computational resources, the ensemble is run in parallel batches, and the dependency is set between consecutive iterations (all batches need to be completed).

---

## Quick Start
To run a new case, duplicate the `case_test` template folder inside the main directory, and modify `config.yaml` as needed. The ensemble size and parallel batch are especially important for cluster systems. The parent folder must contain the following files:

| File | Description |
|---|---|
| `autoencoder_properties.pkl` | VAE architecture configuration |
| `unet_properties.pkl` | UNet architecture configuration |
| `trained_vae.pt` | Pretrained VAE weights |
| `trained_unet.pt` | Pretrained UNet weights |
| `template_folder/` | tNavigator run template (contains `test.DATA`, `KR.INC`, etc.) |

---

## Repository Structure
```
├── config.yaml                   # All workflow parameters
├── submit_all_loop.sh            # Main submission script
├── generate_priors.slurm         # Generate prior geomodels
├── esmda_true.slurm              # Simulate true model
├── esmda_ensemble.slurm          # Run ensemble tNavigator simulations
├── esmda_iter.slurm              # Run ES-MDA iteration
├── tnav_preprocess.slurm         # Preprocess ensemble run folders
├── tnav_preprocess_single.slurm  # Preprocess single (true) run folder
├── tnav_postprocess.slurm        # Postprocess ensemble results
├── tnav_postprocess_single.slurm # Postprocess single (true) results
├── generate_priors.py            # Generate LDM prior samples
├── esmda_iter.py                 # ES-MDA update step
├── tnav_preprocess.py            # Prepare ensemble run folders
├── tnav_preprocess_single.py     # Prepare single (true) run folder
├── tnav_postprocess.py           # Extract ensemble simulation output
├── tnav_postprocess_single.py    # Extract true model simulation output
├── utils.py                      # Shared utilities
└── template_folder/              # tNavigator run template (contains test.DATA, KR.INC, etc.)
```

---

## Configuration
All hardcoded parameters are centralized in `config.yaml`:

| Section | Parameters |
|---|---|
| `paths` | Run directories, diffuser model location |
| `grid` | Spatial dimensions (Nx, Ny, Nz) and latent dimensions (Nx_latent, Ny_latent, Nz_latent) |
| `diffusion` | Scheduler settings and scale factor |
| `facies` | Thresholds, permeability and porosity maps |
| `simulation` | Well counts, time steps, results file path |
| `esmda` | Error variance, alpha schedule, observation times |
| `ensemble` | Ensemble size (`nr`) and ES-MDA iteration count (`Na`) |

---

> **Note:** Resource settings such as memory, walltime, number of CPUs, and GPU allocation are hardcoded in the individual `.slurm` files and should be reviewed and adjusted to match the target cluster and problem size before submission. In particular, `--cpus-per-task` in `esmda_ensemble.slurm` and `esmda_true.slurm` controls the number of threads passed to tNavigator via `OMP_NUM_THREADS`.

## Usage

### Full workflow (including true model simulation)
```bash
./submit_all_loop.sh
```

### Skip true model simulation (use existing data)
```bash
./submit_all_loop.sh 1
```

If skipping the true model, the following file must already exist in `results/` before submission:

| File | Description |
|---|---|
| `all_data_all_times_true.npy` | Simulated production data from the true model, shape `(1, Nt, N_QoI)` where `N_QoI = 2*N_prod + N_inj` (WOPR + WWPR + WWIR) |

---

## Workflow DAG
```
generate_priors
       │
  esmda_true        ← skippable with ./submit_all_loop.sh 1
       │
  tnav_preprocess ─── esmda_ensemble ─── tnav_postprocess ─── esmda_iter
       │                                                            │
      ...                                                          ...
       │                                                            │
  tnav_preprocess ─── esmda_ensemble ─── tnav_postprocess ─── esmda_iter
                                                               (iter Na)
```

---

## Output Files

### `results/`
Files written and updated throughout the workflow:

| File | Written by | Description |
|---|---|---|
| `prior.npy` | `generate_priors.py` | Initial ensemble of geomodels, shape `(nr, Nx, Ny, Nz)` |
| `current.npy` | `generate_priors.py`, `esmda_iter.py` | Current ensemble of geomodels, updated each iteration |
| `z_prior.npy` | `generate_priors.py` | Initial latent vectors, shape `(nr, 1, Nx_latent, Ny_latent, Nz_latent)` |
| `z_current.npy` | `generate_priors.py`, `esmda_iter.py` | Current latent vectors, updated each iteration |
| `csi_prior.npy` | `generate_priors.py` | Initial flattened latents for ES-MDA, shape `(Nx_latent*Ny_latent*Nz_latent, nr)` |
| `csi_current.npy` | `generate_priors.py`, `esmda_iter.py` | Current flattened latents, updated each iteration |
| `all_data_all_times_true.npy` | `tnav_postprocess_single.py` | True model production data, shape `(1, Nt, N_QoI)` |
| `all_data_all_times_iter{i}.npy` | `tnav_postprocess.py` | Ensemble production data at iteration `i`, shape `(nr, Nt, N_QoI)` |
| `current_iter{i}.npy` | `esmda_iter.py` | Snapshot of ensemble geomodels after iteration `i` |
| `csi_current_iter{i}.npy` | `esmda_iter.py` | Snapshot of flattened latents after iteration `i` |
| `z_current_iter{i}.npy` | `esmda_iter.py` | Snapshot of latent vectors after iteration `i` |

### `logs/`
SLURM stdout/stderr logs for each job:

| File pattern | Job |
|---|---|
| `generate_priors_%j.out/err` | Prior generation |
| `true_%j.out/err` | True model simulation |
| `ens_%A_%a.out/err` | Ensemble tNavigator runs (array job) |
| `esmda_iter_%j.out/err` | ES-MDA iteration |
| `tnav_pre_%j.out/err` | Ensemble preprocessing |
| `tnav_pre_single_%j.out/err` | Single model preprocessing |
| `tnav_post_%j.out/err` | Ensemble postprocessing |
| `tnav_post_single_%j.out/err` | Single model postprocessing |
| `out_{task_id}.log` | tNavigator run log per ensemble member |
| `out_true.log` | tNavigator run log for true model |

---

## Requirements
- Python 3.9
- PyTorch
- MONAI Generative
- tNavigator (loaded via HPC module system)
- PyYAML
