# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import sys
import pickle

import yaml
import numpy as np
import torch
import torch.nn as nn

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler

sys.path.append(os.getcwd())
from utils import model2tricat, diffusion_generate_monai_large, z_to_csi

# ─── Config ───────────────────────────────────────────────────────────────────
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# ─── Directories ──────────────────────────────────────────────────────────────
main_dir      = os.getcwd() + '/'
diffuser_dir  = cfg['paths']['diffuser_dir']
results_dir   = os.path.join(main_dir, 'results/')
os.makedirs(results_dir, exist_ok=True)

vae_pickle_path   = os.path.join(diffuser_dir, 'autoencoder_properties.pkl')
unet_pickle_path  = os.path.join(diffuser_dir, 'unet_properties.pkl')
trained_vae_path  = os.path.join(diffuser_dir, 'trained_vae.pt')
trained_unet_path = os.path.join(diffuser_dir, 'trained_unet.pt')

# ─── Grid & Diffusion ─────────────────────────────────────────────────────────
device = 'cuda'

Nx, Ny, Nz                      = cfg['grid']['Nx'],        cfg['grid']['Ny'],        cfg['grid']['Nz']
Nx_latent, Ny_latent, Nz_latent = cfg['grid']['Nx_latent'], cfg['grid']['Ny_latent'], cfg['grid']['Nz_latent']
n_steps      = cfg['diffusion']['n_steps']
scale_factor = cfg['diffusion']['scale_factor']
thresh1      = cfg['facies']['thresh1']
thresh2      = cfg['facies']['thresh2']
nr           = cfg['ensemble']['nr']

# ─── Load VAE ─────────────────────────────────────────────────────────────────
with open(vae_pickle_path, 'rb') as f:
    vae_properties = pickle.load(f)

autoencoder = AutoencoderKL(**vae_properties).to(device)
state_dict  = torch.load(trained_vae_path, map_location=device)
if cfg['diffusion']['multi_gpu']:
    autoencoder = nn.DataParallel(autoencoder)
    autoencoder.load_state_dict(state_dict)
    autoencoder = autoencoder.module
else:
    autoencoder.load_state_dict(state_dict)
autoencoder.eval()


# ─── Load UNet ────────────────────────────────────────────────────────────────
with open(unet_pickle_path, 'rb') as f:
    unet_properties = pickle.load(f)

unet = DiffusionModelUNet(**unet_properties)
    
unet.load_state_dict(torch.load(trained_unet_path))
unet.to(device).eval()

# ─── Scheduler & Inferer ──────────────────────────────────────────────────────
scheduler = DDIMScheduler(
    num_train_timesteps=cfg['diffusion']['n_train_timesteps'],
    schedule=cfg['diffusion']['schedule'],
    beta_start=cfg['diffusion']['beta_start'],
    beta_end=cfg['diffusion']['beta_end'],
    clip_sample=False
)
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# ─── Generate Priors ──────────────────────────────────────────────────────────
print(f"Generating {nr} prior samples...")

z_prior = torch.randn((nr, 1, Nx_latent, Ny_latent, Nz_latent), device=device, dtype=torch.float32)
m_prior = diffusion_generate_monai_large(inferer, unet, autoencoder, scheduler, n_steps, z_prior, n_steps, Nx, Ny, Nz)
m_prior = model2tricat(m_prior, thresh1, thresh2)

# ─── Save ─────────────────────────────────────────────────────────────────────
csi_prior  = z_to_csi(z_prior[:, 0].cpu().numpy())
z_prior_np = z_prior.cpu().numpy()

for fname, arr in [
    ('prior.npy',       m_prior),
    ('current.npy',     m_prior),
    ('csi_prior.npy',   csi_prior),
    ('csi_current.npy', csi_prior),
    ('z_prior.npy',     z_prior_np),
    ('z_current.npy',   z_prior_np),
]:
    np.save(os.path.join(results_dir, fname), arr)
    print(f"Saved {fname}")
