'''
File: train_unet.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to train a U-net to learn the de-noising process in the latent space of latent diffusion models
'''

# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import h5py
import pickle

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler

from utils import (
    model2tricat,
    build_hard_data_pickle,
    save_hard_data_pickle,
    create_dataloader,
    print_losses
)

# ─── Config ───────────────────────────────────────────────────────────────────
with open('config_unet.yaml') as f:
    cfg = yaml.safe_load(f)

# ─── Directories ──────────────────────────────────────────────────────────────
case_dir          = cfg['paths']['case_dir']
h5_file_path      = cfg['paths']['h5_file']

unet_dir          = os.path.join(case_dir, cfg['paths']['unet_dir'])
vae_dir           = os.path.join(case_dir, cfg['paths']['vae_dir'])
os.makedirs(case_dir, exist_ok=True)
os.makedirs(unet_dir, exist_ok=True)

trained_unet_path = os.path.join(unet_dir, 'trained_unet')
unet_pickle_path  = os.path.join(unet_dir, 'unet_properties.pkl')
unet_txt_path     = os.path.join(unet_dir, 'unet_properties.txt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── Training Settings ────────────────────────────────────────────────────────
seed          = cfg['training']['seed']
n_epochs      = cfg['training']['n_epochs']
batch_size    = cfg['training']['batch_size']
val_interval  = cfg['training']['val_interval']
save_interval = cfg['training']['save_interval']
lr            = float(cfg['training']['lr'])
multi_gpu_vae = cfg['vae']['multi_gpu']

np.random.seed(seed)
torch.manual_seed(seed)

# ─── Load Data ────────────────────────────────────────────────────────────────
with h5py.File(h5_file_path, 'r') as f:
    models_loaded = model2tricat(np.array(f["data"])[:] / 255.,
                                 cfg['facies']['thresh1'],
                                 cfg['facies']['thresh2'])

indices = np.arange(models_loaded.shape[0])
np.random.shuffle(indices)
models_loaded = models_loaded[indices]

# ─── Hard Data ────────────────────────────────────────────────────────────────
well_loc = {name: tuple(coords) for name, coords in cfg['wells'].items()}

hard_data_dict   = build_hard_data_pickle(models_loaded, well_loc)
save_hard_data_pickle(hard_data_dict, case_dir)
well_hd_combined = np.concatenate(list(hard_data_dict.values()), axis=0)

# ─── Dataloaders ──────────────────────────────────────────────────────────────
train_end = cfg['data_split']['train_end']
val_start = cfg['data_split']['val_start']
val_end   = cfg['data_split']['val_end']
test_end  = cfg['data_split']['test_end']

train_loader = create_dataloader(models_loaded[:train_end],        batch_size, shuffle=True)
val_loader   = create_dataloader(models_loaded[val_start:val_end], batch_size, shuffle=False)
test_loader  = create_dataloader(models_loaded[val_end:test_end],  batch_size, shuffle=False)

# ─── Load Pretrained VAE ──────────────────────────────────────────────────────
vae_epoch        = cfg['vae']['epoch']
scale_factor     = cfg['vae']['scale_factor']
trained_vae_path = os.path.join(vae_dir, f"trained_vae{vae_epoch}.pt")
vae_pickle_path  = os.path.join(vae_dir, "autoencoder_properties.pkl")

with open(vae_pickle_path, 'rb') as f:
    vae_properties = pickle.load(f)

autoencoder = AutoencoderKL(**vae_properties).to(device)
state_dict  = torch.load(trained_vae_path, map_location=device)
if multi_gpu_vae:
    autoencoder = torch.nn.DataParallel(autoencoder)
    autoencoder.load_state_dict(state_dict)
    autoencoder = autoencoder.module
else:
    autoencoder.load_state_dict(state_dict)
autoencoder.eval()



# ─── UNet ─────────────────────────────────────────────────────────────────────
unet_properties = {
    'spatial_dims':      cfg['unet']['spatial_dims'],
    'in_channels':       cfg['unet']['in_channels'],
    'out_channels':      cfg['unet']['out_channels'],
    'num_channels':      tuple(cfg['unet']['num_channels']),
    'num_res_blocks':    cfg['unet']['num_res_blocks'],
    'attention_levels':  tuple(cfg['unet']['attention_levels']),
    'num_head_channels': tuple(cfg['unet']['num_head_channels']),
}

unet           = DiffusionModelUNet(**unet_properties).to(device)
optimizer_diff = torch.optim.Adam(unet.parameters(), lr=lr)
scaler         = GradScaler()

with open(unet_pickle_path, 'wb') as f:
    pickle.dump(unet_properties, f)

with open(unet_txt_path, 'w') as f:
    f.write("\n".join(f"{k}: {v}" for k, v in unet_properties.items()))

# ─── Scheduler & Inferer ──────────────────────────────────────────────────────
scheduler = DDPMScheduler(
    num_train_timesteps=cfg['diffusion']['n_train_timesteps'],
    schedule=cfg['diffusion']['schedule'],
    beta_start=cfg['diffusion']['beta_start'],
    beta_end=cfg['diffusion']['beta_end'],
)
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# ─── Loss Tracking ────────────────────────────────────────────────────────────
epoch_loss_list = []
val_loss_list   = []

# ─── Training Loop ────────────────────────────────────────────────────────────
Nx_latent = cfg['grid']['Nx_latent']
Ny_latent = cfg['grid']['Ny_latent']
Nz_latent = cfg['grid']['Nz_latent']

for epoch in range(n_epochs):

    unet.train()
    train_loss = 0.0

    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
    train_loop.set_description(f"Epoch {epoch + 1}/{n_epochs}")

    for step, batch in train_loop:
        images = batch["image"].to(device)
        optimizer_diff.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            latent    = torch.randn((images.shape[0], 1, Nx_latent, Ny_latent, Nz_latent)).to(device)
            noise     = torch.randn_like(latent)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (images.shape[0],), device=device
            ).long()

            noise_pred = inferer(
                inputs=images,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                noise=noise,
                timesteps=timesteps,
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer_diff)
        scaler.update()

        train_loss += loss.item()
        train_loop.set_postfix({"loss": f"{train_loss / (step + 1):.6f}"})

    epoch_loss_list.append(train_loss / len(train_loader))
    print_losses(
        split="Train", epoch=epoch + 1, n_epochs=n_epochs,
        loss=train_loss / len(train_loader),
    )

    # ─── Validation ───────────────────────────────────────────────────────────
    if (epoch + 1) % val_interval == 0:
        unet.eval()
        val_loss = 0.0

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                with autocast(enabled=True):
                    latent    = torch.randn((images.shape[0], 1, Nx_latent, Ny_latent, Nz_latent)).to(device)
                    noise     = torch.randn_like(latent)
                    timesteps = torch.randint(
                        0, scheduler.num_train_timesteps, (images.shape[0],), device=device
                    ).long()
                    noise_pred = inferer(
                        inputs=images,
                        autoencoder_model=autoencoder,
                        diffusion_model=unet,
                        noise=noise,
                        timesteps=timesteps,
                    )
                    val_loss += F.mse_loss(noise_pred.float(), noise.float()).item()

        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)
        print_losses(
            split="Val", epoch=epoch + 1, n_epochs=n_epochs,
            loss=val_loss,
        )

    # ─── Checkpoint ───────────────────────────────────────────────────────────
    if (epoch + 1) % save_interval == 0:
        torch.save(unet.state_dict(), f"{trained_unet_path}{epoch + 1}.pt")
        print(f"Model saved to {trained_unet_path}{epoch + 1}.pt")

# ─── Save Final Model ─────────────────────────────────────────────────────────
torch.save(unet.state_dict(), f"{trained_unet_path}.pt")
print(f"Final model saved to {trained_unet_path}.pt")
