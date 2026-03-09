# Multiscenario 3D Latent Diffusion for Geomodel Parameterization

Code example for "3D latent diffusion models for parameterizing and history matching multiscenario facies systems" [https://link.springer.com/article/10.1007/s11004-025-10245-x]

## Summary
We present a deep-learning geological parameterization for complex facies-based geomodels, using recently developed generative latent diffusion models (LDMs), first published by Rombach et al. (2022). Diffusion models are trained to ''denoise'', which enables them to generate new geological realizations from input fields characterized by random noise. Based on Denoising Probabilstic Diffusion Models (DDPMs), introduced by Ho et al. (2020), the LDM representation reduces the number of variables to be determined during history matching, while preserving realistic geological features in posterior models. The model developed in this work includes a variational autoencoder (VAE) for dimension reduction, a U-net for the denoising process, and a Denoising Diffusion Implicit Model (DDIM, Song et al. (2021)) noise scheduling for inference. Additionally, a perceptual (style) loss is used to improve the geological realism/visual quality of generated models. A dimension reduction ratio of 512 is achieved between geomodel and latent space. New geomodels can be generated in a fraction of the computational time required for geostatistical simulation (object based modeling)

Our application involves large, conditional 3D three-facies (channel-levee-mud) systems, with uncertain geological scenario. Geological scenario parameters include mud fraction, channel orientation and width. The LDM can provide realizations that are visually consistent with samples from geomodeling software. General agreement between the diffusion-generated models and reference realizations can be observed through quantitative metrics involving spatial and flow-response statistics. Ranges and distributions of geological scenario parameters are also consistent between reference and LDM-generated geomodels. The LDM can then be used for ensemble-based data assimilation, in an uncertain geological scenario setting. Significant uncertainty reduction, posterior P10-P90 forecasts that generally bracket observed data, and consistent posterior geomodels with correct geological scenario parameters, can be achieved.


## Contents
- `scripts/` - Directory with `.py` scripts to train the VAE and U-net and generate new realizations with the 3D-LDM. Contains the following files:
  - `train_vae.py` — trains the variational autoencoder
  - `train_unet.py` — trains the U-net denoising model in latent space
  - `generate.py` — generates new geomodel realizations using the trained LDM
  - `utils.py` — shared utility functions for data processing, latent space operations, and diffusion generation

- `case_test/` - Template case folder to be duplicated for new runs. Contains:
  - `config_vae.yaml` — all parameters for VAE training
  - `config_unet.yaml` — all parameters for U-net training and generation

- `esmda/` - Directory with scripts for ensemble-based history matching (ES-MDA) using the trained LDM as a geological parameterization. Contains SLURM submission scripts and Python scripts for the ES-MDA workflow. See the dedicated `README.md for more details.

- `data/` - Directory to store the training dataset (3D, three-facies multiple scenario channelized geomodels) and synthetic true models used in history matching (`true.npy`). Data files are available at the link provided.


## Quick Start
To set up a new case:

1. Duplicate `case_test/` and rename it for your case.
2. Edit `config_vae.yaml` and `config_unet.yaml` — at minimum update `h5_file` to point to your dataset. Set `case_dir: "./"` to write all outputs to the case folder.
3. Train the VAE (run from inside the case folder):
```bash
python ../scripts/train_vae.py
```
Checkpoints will be saved to `trained_vae/`. The scale factor will be automatically computed and written to `config_unet.yaml`.

4. Train the U-net (update `vae.epoch` in `config_unet.yaml` to select which VAE checkpoint to load; leave empty for the final checkpoint):
```bash
python ../scripts/train_unet.py
```
Checkpoints will be saved to `trained_unet/`.

5. Generate new samples (update `vae_epoch` and `unet_epoch` in `generate.py` to select checkpoints; leave empty for the final checkpoint):
```bash
python ../scripts/generate.py
```
Generated samples will be saved to `generated_samples.npy`.

## Configuration
All training and generation parameters are managed through two YAML configuration files located in the case folder:
- `config_vae.yaml` — paths, grid dimensions, facies thresholds, well locations, data split, VAE architecture, discriminator, training settings, and loss weights.
- `config_unet.yaml` — paths, grid dimensions, facies thresholds, well locations, data split, VAE checkpoint, UNet architecture, training settings, diffusion scheduler parameters, and VAE scale factor (written automatically by `train_vae.py`).

Trained model checkpoints are saved in subdirectories `trained_vae/` and `trained_unet/` within the case folder.

## Software Requirements
Running the scripts requires the following Python libraries: `torch`, `monai-generative`, `h5py`, `numpy`, `pandas`, `pyyaml`, and `tqdm`.
\
This workflow is tested with Python 3.9 and PyTorch 1.8 (CPU/GPU). GPU is strongly recommended for training.

## Code implementations are based on the following repositories:
- [diffusers](https://github.com/huggingface/diffusers/)
- [monai](https://github.com/Project-MONAI/tutorials/tree/main/generative) and [tutorial](https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/3d_ldm/3d_ldm_tutorial.ipynb)

## Contact
Guido Di Federico, Louis J. Durlofsky  
Department of Energy Science & Engineering, Stanford University  
Contact: gdifede@stanford.edu

## Examples
<figure style="text-align: center; margin-bottom: 100px;">
  <img src="./pics/vae_training.jpg?raw=true" alt="Alt text" title="Title" width="500"/>
  <figcaption>VAE training procedure</figcaption>
</figure>

<figure style="text-align: center; margin-bottom: 100px;">
  <img src="./pics/unet_training.jpg?raw=true" alt="Alt text" title="Title" width="500"/>
  <figcaption>U-net training procedure</figcaption>
</figure>

<figure style="text-align: center; margin-bottom: 100px;">
  <img src="./pics/ldm_generation.jpg?raw=true" alt="Alt text" title="Title" width="500"/>
  <figcaption>3D-LDM generation of a new geomodel realization</figcaption>
</figure>

<figure style="text-align: center; margin-bottom: 100px;">
  <img src="./pics/denoising_3d.jpg?raw=true" alt="Alt text" title="Title" width="500"/>
  <figcaption>Visualization of 3D denoising with variable number of steps</figcaption>
</figure>
