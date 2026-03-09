# Multiscenario 3D Latent Diffusion for Geomodel Parameterization

Code example for "3D latent diffusion models for parameterizing and history matching multiscenario facies systems" [https://link.springer.com/article/10.1007/s11004-025-10245-x]

## Summary
We present a deep-learning geological parameterization for complex facies-based geomodels, using recently developed generative latent diffusion models (LDMs), first published by Rombach et al. (2022). Diffusion models are trained to ''denoise'', which enables them to generate new geological realizations from input fields characterized by random noise. Based on Denoising Probabilstic Diffusion Models (DDPMs), introduced by Ho et al. (2020), the LDM representation reduces the number of variables to be determined during history matching, while preserving realistic geological features in posterior models. The model developed in this work includes a variational autoencoder (VAE) for dimension reduction, a U-net for the denoising process, and a Denoising Diffusion Implicit Model (DDIM, Song et al. (2021)) noise scheduling for inference. Additionally, a perceptual (style) loss is used to improve the geological realism/visual quality of generated models. A dimension reduction ratio of 512 is achieved between geomodel and latent space. New geomodels can be generated in a fraction of the computational time required for geostatistical simulation (object based modeling)

Our application involves large, conditional 3D three-facies (channel-levee-mud) systems, with uncertain geological scenario. Geological scenario parameters include mud fraction, channel orientation and width. The LDM can provide realizations that are visually consistent with samples from geomodeling software. General agreement between the diffusion-generated models and reference realizations can be observed through quantitative metrics involving spatial and flow-response statistics. Ranges and distributions of geological scenario parameters are also consistent between reference and LDM-generated geomodels. The LDM can then be used for ensemble-based data assimilation, in an uncertain geological scenario setting. Significant uncertainty reduction, posterior P10-P90 forecasts that generally bracket observed data, and consistent posterior geomodels with correct geological scenario parameters, can be achieved.


## Contents
- `scripts/` - Directory with `.py` scripts to preprocess the dataset, train the variational autoencoder (VAE) and U-net, and generate new realizations with 3D-LDM. It also contains the trained VAE and U-net.
- `data/` - Directory to store training dataset used in this study `m_petrel.h5`  (3D, three-facies multiple scenario channelized geomodels). Additional files are the LDM-generated ensemble `m_gen_ldm.npy` used for static and flow statistics as well as for priors in history matching. Files are stored as `.h5` at the link provided. Synthetic "true" models used in history matching are saved as `m_true_1.npy`,  `m_true_2.npy` and `m_true_3.npy`. All are stored as `.npy` files.

## Quick Start
To set up a new case:

1. Duplicate the template folder and rename it for your case.
2. Edit `config_vae.yaml` and `config_unet.yaml` — at minimum update `case_dir` and `h5_file` to point to your case directory and dataset.
3. Train the VAE:
```bash
   python train_vae.py
```
   Checkpoints will be saved to `<<case_dir>>/trained_vae/`.

4. Train the U-net (update `vae_epoch` in `config_unet.yaml` to select which VAE checkpoint to load; set to `""` for the final step):
```bash
   python train_unet.py
```
   Checkpoints will be saved to `<<case_dir>>/trained_unet/`.

5. Generate new samples (update `vae_epoch` and `unet_epoch` in `generate.py` to select checkpoints; set to `""` for the final step):
```bash
   python generate.py
```
   Generated samples will be saved to `<<case_dir>>/generated_samples.npy`.

## Configuration
All training and generation parameters are managed through two YAML configuration files:
- `config_vae.yaml` - Paths, grid dimensions, facies thresholds, well locations, data split, VAE architecture, discriminator, training settings, loss weights, and diffusion scheduler parameters for VAE training.
- `config_unet.yaml` - Paths, grid dimensions, facies thresholds, well locations, data split, VAE checkpoint, UNet architecture, training settings, and diffusion scheduler parameters for UNet training and generation.

Trained model checkpoints are saved in subdirectories `trained_vae/` and `trained_unet/` within `case_dir`.

## Code implementations are based on the following repositories:
- [diffusers](https://github.com/huggingface/diffusers/)
- [monai](https://github.com/Project-MONAI/tutorials/tree/main/generative) and [tutorial](https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/3d_ldm/3d_ldm_tutorial.ipynb)

## Software requirements
Running the scripts requires the libraries `datasets`, `diffusers`, `monai` or `monai-generative`, and `pyyaml`.
\
This workflow is tested with Python 3.9 and PyTorch 1.8 (CPU/GPU).

## Contact
Guido Di Federico, Louis J. Durlofsky  
Department of Energy Science & Engineering, Stanford University 
\
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
