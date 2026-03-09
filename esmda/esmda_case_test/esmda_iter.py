# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import sys
import pickle
import argparse

import yaml
import numpy as np
import torch
import torch.nn as nn

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler

sys.path.append(os.getcwd())
from utils import model2tricat, diffusion_generate_monai_large, csi_to_z

# ─── Config ───────────────────────────────────────────────────────────────────
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run ES-MDA iteration by index.")
parser.add_argument('--index', type=int, required=True, help='ES-MDA iteration index')
args = parser.parse_args()
i_na = args.index

# ─── Directories ──────────────────────────────────────────────────────────────
main_dir      = os.getcwd() + '/'
diffuser_dir  = cfg['paths']['diffuser_dir']
results_dir   = os.path.join(main_dir, 'results/')
os.makedirs(results_dir, exist_ok=True)

vae_pickle_path  = os.path.join(diffuser_dir, 'autoencoder_properties.pkl')
unet_pickle_path = os.path.join(diffuser_dir, 'unet_properties.pkl')
trained_vae_path = os.path.join(diffuser_dir, 'trained_vae.pt')
trained_unet_path = os.path.join(diffuser_dir, 'trained_unet.pt')

# ─── Grid & Diffusion ─────────────────────────────────────────────────────────
device = 'cuda'

Nx, Ny, Nz                      = cfg['grid']['Nx'],       cfg['grid']['Ny'],       cfg['grid']['Nz']
Nx_latent, Ny_latent, Nz_latent = cfg['grid']['Nx_latent'], cfg['grid']['Ny_latent'], cfg['grid']['Nz_latent']
n_steps      = cfg['diffusion']['n_steps']
scale_factor = cfg['diffusion']['scale_factor']
thresh1      = cfg['facies']['thresh1']
thresh2      = cfg['facies']['thresh2']

# ─── ES-MDA Parameters ────────────────────────────────────────────────────────
error_q = cfg['esmda']['error_q']
alpha   = np.array(cfg['esmda']['alpha'])
times   = cfg['esmda']['times']
na      = len(alpha)
nr      = cfg['ensemble']['nr']

# Early exit if all iterations complete
if i_na == na:
    print(f"All {na} ES-MDA iterations complete. Exiting.")
    exit()

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

# ─── Load Simulation Data ─────────────────────────────────────────────────────
print(f"Running ES-MDA iteration {i_na}/{na}...")

all_data_true = np.load(os.path.join(results_dir, 'all_data_all_times_true.npy'))[0][:len(times)]
all_data      = np.load(os.path.join(results_dir, f'all_data_all_times_iter{i_na}.npy'))[:, :len(times)]

# ─── Construct Observed Data ──────────────────────────────────────────────────
d_obs = all_data_true.flatten(order='F').reshape(-1, 1)
nd    = len(d_obs)
cd    = np.diag((error_q * d_obs.flatten()) ** 2)

# ─── ES-MDA Update ────────────────────────────────────────────────────────────
d_uc = np.zeros((nd, nr))
for i in range(nr):
    d_uc[:, i] = (d_obs + np.random.multivariate_normal(np.zeros(nd), alpha[i_na] * cd, 1).T).reshape(-1)

d_k   = all_data.T.reshape((nd, nr), order='C')
csi_k = np.load(os.path.join(results_dir, 'csi_current.npy'))

ones    = np.ones((1, nr))
csi_ave = np.mean(csi_k, axis=1, keepdims=True)
d_ave   = np.mean(d_k,   axis=1, keepdims=True)

cmd    = (csi_k - csi_ave @ ones) @ (d_k - d_ave @ ones).T / (nr - 1)
cdd    = (d_k   - d_ave   @ ones) @ (d_k - d_ave @ ones).T / (nr - 1)
cd_inv = np.linalg.inv(cdd + alpha[i_na] * cd)

gain = cmd @ cd_inv
for i in range(nr):
    csi_k[:, i] += gain @ (d_uc[:, i] - d_k[:, i])

# ─── Generate Updated Models ──────────────────────────────────────────────────
z_k = torch.tensor(
    csi_to_z(csi_k, Nx_latent, Ny_latent, Nz_latent)[:, np.newaxis],
    dtype=torch.float32, device=device
)
m_k = diffusion_generate_monai_large(inferer, unet, autoencoder, scheduler, n_steps, z_k, n_steps, Nx, Ny, Nz)
m_k = model2tricat(m_k, thresh1, thresh2)

# ─── Save ─────────────────────────────────────────────────────────────────────
z_k_np = z_k.cpu().numpy()

for fname, arr in [
    ('current.npy',                 m_k),
    ('csi_current.npy',             csi_k),
    (f'current_iter{i_na}.npy',     m_k),
    (f'csi_current_iter{i_na}.npy', csi_k),
    ('z_current.npy',               z_k_np),
    (f'z_current_iter{i_na}.npy',   z_k_np),
]:
    np.save(os.path.join(results_dir, fname), arr)

print(f"Iteration {i_na} complete. Results saved to {results_dir}")
