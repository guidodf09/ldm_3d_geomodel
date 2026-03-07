'''
File: generate.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to generate new samples with a trained VAE and U-net (i.e., latent diffusion model)
'''

# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import pickle

import yaml
import numpy as np
import torch

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler

from utils import model2tricat, diffusion_generate_monai_large, load_hard_data_pickle, compute_hd_accuracy

# ─── Config ───────────────────────────────────────────────────────────────────
with open('config_unet.yaml') as f:
    cfg = yaml.safe_load(f)

# ─── Generation Parameters ────────────────────────────────────────────────────
n_samples    = 100
n_steps      = 100
num_per_iter = 10
vae_epoch    = ''
unet_epoch   = ''

# ─── From Config ──────────────────────────────────────────────────────────────
case_dir                        = cfg['paths']['case_dir']
vae_dir                         = os.path.join(case_dir, cfg['paths']['vae_dir'])
unet_dir                        = os.path.join(case_dir, cfg['paths']['unet_dir'])
Nx, Ny, Nz                      = cfg['grid']['Nx'],        cfg['grid']['Ny'],        cfg['grid']['Nz']
Nx_latent, Ny_latent, Nz_latent = cfg['grid']['Nx_latent'], cfg['grid']['Ny_latent'], cfg['grid']['Nz_latent']
thresh1                         = cfg['facies']['thresh1']
thresh2                         = cfg['facies']['thresh2']
multi_gpu                       = cfg['vae']['multi_gpu']
scale_factor                    = cfg['vae']['scale_factor']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── Load Pretrained VAE ──────────────────────────────────────────────────────
trained_vae_path = os.path.join(vae_dir, f"trained_vae{vae_epoch}.pt")
vae_pickle_path  = os.path.join(vae_dir, "autoencoder_properties.pkl")

with open(vae_pickle_path, 'rb') as f:
    vae_properties = pickle.load(f)

autoencoder = AutoencoderKL(**vae_properties).to(device)
state_dict  = torch.load(trained_vae_path, map_location=device)
if multi_gpu:
    autoencoder = torch.nn.DataParallel(autoencoder)
    autoencoder.load_state_dict(state_dict)
    autoencoder = autoencoder.module
else:
    autoencoder.load_state_dict(state_dict)
autoencoder.eval()

# ─── Load Pretrained UNet ─────────────────────────────────────────────────────
trained_unet_path = os.path.join(unet_dir, f"trained_unet{unet_epoch}.pt")
unet_pickle_path  = os.path.join(unet_dir, "unet_properties.pkl")

with open(unet_pickle_path, 'rb') as f:
    unet_properties = pickle.load(f)

unet = DiffusionModelUNet(**unet_properties).to(device)
unet.load_state_dict(torch.load(trained_unet_path, map_location=device))
unet.eval()

# ─── Scheduler & Inferer ──────────────────────────────────────────────────────
scheduler = DDIMScheduler(
    num_train_timesteps=cfg['diffusion']['n_train_timesteps'],
    schedule=cfg['diffusion']['schedule'],
    beta_start=cfg['diffusion']['beta_start'],
    beta_end=cfg['diffusion']['beta_end'],
    clip_sample=False,
)
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# ─── Generate ─────────────────────────────────────────────────────────────────
print(f"Generating {n_samples} samples...")

noise = torch.randn((n_samples, 1, Nx_latent, Ny_latent, Nz_latent)).to(device)
with torch.no_grad():
    generated_samples = diffusion_generate_monai_large(
        inferer=inferer,
        unet=unet,
        vae=autoencoder,
        scheduler=scheduler,
        latent=noise,
        n_steps=n_steps,
        num_per_iter=num_per_iter,
        nx=Nx,
        ny=Ny,
        nz=Nz,
    )

generated_samples_tricat = model2tricat(generated_samples, thresh1, thresh2)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(case_dir, 'generated_samples.npy')
np.save(out_path, generated_samples_tricat)
print(f"Generated {n_samples} samples saved to {out_path}")
