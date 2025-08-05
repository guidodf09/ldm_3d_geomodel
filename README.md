# Multiscenario 3D Latent Diffusion for Geomodel Parameterization

Code example for "3D latent diffusion models for parameterizing and history matching multiscenario facies systems"
## Summary
We present a deep-learning geological parameterization for complex facies-based geomodels, using recently developed generative latent diffusion models (LDMs), first published by Rombach et al. (2022). Diffusion models are trained to ''denoise'', which enables them to generate new geological realizations from input fields characterized by random noise. Based on Denoising Probabilstic Diffusion Models (DDPMs), introduced by Ho et al. (2020), the LDM representation reduces the number of variables to be determined during history matching, while preserving realistic geological features in posterior models. The model developed in this work includes a variational autoencoder (VAE) for dimension reduction, a U-net for the denoising process, and a Denoising Diffusion Implicit Model (DDIM, Song et al. (2021)) noise scheduling for inference. Additionally, a perceptual (style) loss is used to improve the geological realism/visual quality of generated models. A dimension reduction ratio of 512 is achieved between geomodel and latent space. New geomodels can be generated in a fraction of the computational time required for geostatistical simulation (object based modeling)

Our application involves large, conditional 3D three-facies (channel-levee-mud) systems, with uncertain geological scenario. Geological scenario parameters include mud fraction, channel orientation and width. The LDM can provide realizations that are visually consistent with samples from geomodeling software. General agreement between the diffusion-generated models and reference realizations can be observed through quantitative metrics involving spatial and flow-response statistics. Ranges and distributions of geological scenario parameters are also consistent between reference and LDM-generated geomodels. The LDM can then be used for ensemble-based data assimilation, in an uncertain geological scenario setting. Significant uncertainty reduction, posterior P10-P90 forecasts that generally bracket observed data, and consistent posterior geomodels with correct geological scenario parameters, can be achieved. 
...
## Contents
- `scripts/` - Directory to store dataset for data preparation, variational autoencoder (VAE) training and U-net training `.py` scripts.
- `data/` - Directory to store training dataset used in this study (2D, three-facies channelized geomodels). Dataset is stored as datasets.Dataset folder (`diffusers_dataset/`) 
- `testing/` - Directory to store reference (geomodeling software-generated) `m_petrel_200.npy` and LDM-generated `m_ldm_200.npy` ensembles used for flow simulations and history matching. Synthetic "true" models used in history matching are saved as `m_true_1.npy`,  `m_true_2.npy` and `m_true_3.npy`. All are stored as `.npy` files.

Code implementations are based on the following repositories:
- [diffusers](https://github.com/huggingface/diffusers/)
- [monai](https://github.com/Project-MONAI/tutorials/tree/main/generative)

## Software requirements
Running the scripts requires the libraries `datasets`,  `diffusers`,  `monai` or  `monai-generative`.
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

