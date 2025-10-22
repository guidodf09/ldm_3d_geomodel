
import os
import re
import psutil
import h5py
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import MSELoss
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D

from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import Dataset
from monai.utils import first, set_determinism

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import datasets
import shutil
import tempfile
import torch.nn.functional as F


from utils import (
    KL_loss,
    model2tricat,
    build_hard_data_pickle,
    save_hard_data_pickle,
    load_hard_data_pickle
)

# ---------------------------- Configs ---------------------------- #
main_dir = './'
case_dir = os.path.join(main_dir, 'scripts/')
os.makedirs(case_dir, exist_ok=True)

h5_file_path = '../data/geomodels_128_paper.h5'

# -------------------------- Load Data --------------------------- #
with h5py.File(h5_file_path, 'r') as f:
    models_loaded = model2tricat(np.array(f["data"])[:] / 255., 0.25, 0.8)

np.random.shuffle(models_loaded)

nx, ny, nz = models_loaded.shape[2:]
size = [nx, ny, nz]

well_loc = {
    'i1': (16, 16), 'i2': (16, 64), 'i3': (16, 112), 'i4': (64, 16),
    'p1': (64, 64), 'p2': (64, 112), 'p3': (112, 16), 'p4': (112, 64), 'p5': (112, 112)
}

hard_data_dict = build_hard_data_pickle(models_loaded, well_loc)
save_hard_data_pickle(hard_data_dict, case_dir)

# Load back combined hard data points as single array
well_hd_all = load_hard_data_pickle(case_dir)
well_hd_combined = np.concatenate(list(well_hd_all.values()), axis=0)

# ------------------------ Dataset Setup ------------------------- #
train_datalist = [{"image": torch.tensor(models_loaded[i])} for i in range(0, 2400)]
train_ds = Dataset(data=train_datalist)
train_loader = DataLoader(train_ds, batch_size=2)

val_datalist = [{"image": torch.tensor(models_loaded[i])} for i in range(2400, 2700)]
val_ds = Dataset(data=val_datalist)
val_loader = DataLoader(val_ds, batch_size=2)

# ------------------------ Model Setup --------------------------- #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae_epoch = 1000
trained_vae_path = os.path.join(case_dir, f'trained_vae_{vae_epoch}_maybewrong.pt')
vae_pickle_path = os.path.join(case_dir, 'autoencoder_properties.pkl')

with open(vae_pickle_path, 'rb') as pkl_file:
    vae_properties = pickle.load(pkl_file)

autoencoder = AutoencoderKL(
    spatial_dims=vae_properties['spatial_dims'],
    in_channels=vae_properties['in_channels'],
    out_channels=vae_properties['out_channels'],
    num_channels=vae_properties['num_channels'],
    latent_channels=vae_properties['latent_channels'],
    num_res_blocks=vae_properties['num_res_blocks'],
    norm_num_groups=vae_properties['norm_num_groups'],
    attention_levels=vae_properties['attention_levels'],
)

autoencoder.to(device)

state_dict = torch.load(trained_vae_path)
autoencoder.load_state_dict(state_dict)
autoencoder.eval()

trained_unet_path = os.path.join(case_dir, 'trained_unet')
unet_pickle_path = os.path.join(case_dir, 'unet_properties.pkl')
unet_txt_path = os.path.join(case_dir, 'unet_properties.txt')

unet_properties = {
    'spatial_dims': 3,
    'in_channels': 1,
    'out_channels': 1,
    'num_channels': (64, 128, 256),
    'num_res_blocks': 1,
    'attention_levels': (False, True, True),
    'num_head_channels': (0, 128, 256),
}

with open(unet_pickle_path, 'wb') as file:
    pickle.dump(unet_properties, file)

properties_str = "\n".join(f"{key}: {value}" for key, value in unet_properties.items())

with open(unet_txt_path, 'w') as txt_file:
    txt_file.write(properties_str)

unet = DiffusionModelUNet(
    spatial_dims=unet_properties['spatial_dims'],
    in_channels=unet_properties['in_channels'],
    out_channels=unet_properties['out_channels'],
    num_channels=unet_properties['num_channels'],
    num_res_blocks=unet_properties['num_res_blocks'],
    attention_levels=unet_properties['attention_levels'],
    num_head_channels=unet_properties['num_head_channels'],
)

unet.to(device)

scale_factor = 1
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)

n_epochs = 1000
epoch_loss_list = []
autoencoder.eval()
scaler = GradScaler()

n_epochs = 1000
epoch_loss_list = []

for epoch in range(n_epochs):
    if (epoch + 1) % 500 == 0:
        torch.save(unet.state_dict(), f'{trained_unet_path}_{epoch + 1}_ddpm_maybewrong.pt')

    # --- Training ---
    unet.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer_diff.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            z = torch.randn((images.shape[0], 1, 16, 16, 4), device=device)
            noise = torch.randn_like(z)

            timesteps = torch.randint(
                0,
                inferer.scheduler.num_train_timesteps,
                (images.shape[0],),
                device=device
            ).long()

            noise_pred = inferer(
                inputs=images,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                noise=noise,
                timesteps=timesteps
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer_diff)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"train_loss": epoch_loss / (step + 1)})

    epoch_loss_list.append(epoch_loss / len(train_loader))

    # --- Validation ---
    unet.eval()
    val_loss = 0.0

    with torch.no_grad():
        for val_step, batch in enumerate(val_loader):
            images = batch["image"].to(device)

            with autocast(enabled=True):
                z = torch.randn((images.shape[0], 1, 16, 16, 4), device=device)
                noise = torch.randn_like(z)

                timesteps = torch.randint(
                    0,
                    inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=device
                ).long()

                noise_pred = inferer(
                    inputs=images,
                    autoencoder_model=autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps
                )

                val_loss_batch = F.mse_loss(noise_pred.float(), noise.float())

            val_loss += val_loss_batch.item()

    val_loss /= len(val_loader)
    print(f"\n[Validation] Epoch {epoch}: Validation Loss: {val_loss:.6f}")
