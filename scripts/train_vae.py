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

vae_properties = {
    'spatial_dims': 3,
    'in_channels': 1,
    'out_channels': 1,
    'num_channels': (64, 128, 256, 512),
    'latent_channels': 1,
    'num_res_blocks': 1,
    'norm_num_groups': 16,
    'attention_levels': (False, False, False, True)
}

autoencoder = AutoencoderKL(**vae_properties).to(device)
optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)

# Losses
l2_loss = MSELoss()
loss_perceptual = PerceptualLoss(
    spatial_dims=3,
    network_type="squeeze",
    is_fake_3d=True,
    fake_3d_ratio=0.2
).to(device)

# ---------------------- Loss Weights & Paths --------------------- #
n_epochs = 1000
perceptual_weight = 0.001
kl_weight = 1e-6
hd_weight = 1e-2

trained_vae_path = os.path.join(case_dir, 'trained_vae')

# ---------------------- Loss Tracking Lists ---------------------- #
epoch_recon_loss_list, epoch_kl_loss_list = [], []
epoch_percep_loss_list, epoch_hd_loss_list = [], []

val_recon_loss_list, val_kl_loss_list = [], []
val_percep_loss_list, val_hd_loss_list = [], []

# ---------------------- HD Loss Function ------------------------ #
def compute_hd_loss(y_pred, well_hd_all):
    ix = list(well_hd_all[:, 0].astype(int))
    iy = list(well_hd_all[:, 1].astype(int))
    iz = list(well_hd_all[:, 2].astype(int))
    v = torch.from_numpy(well_hd_all[:, -1]).float().to(device).repeat(y_pred.shape[0], 1)
    preds = y_pred[:, 0, ix, iy, iz]
    return F.mse_loss(preds, v)

# --------------------------- Training --------------------------- #
for epoch in range(n_epochs):
    if (epoch + 1) % 50 == 0:
        torch.save(autoencoder.state_dict(), f'{trained_vae_path}_{epoch + 1}.pt')

    autoencoder.train()
    epoch_recons_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_percep_loss = 0.0
    epoch_hd_loss = 0.0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = batch["image"].to(device)

        optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = autoencoder(images)

        kl = KL_loss(z_mu, z_sigma)
        recons = l2_loss(reconstruction.float(), images.float())
        perceptual = loss_perceptual(reconstruction.float(), images.float())
        hd = compute_hd_loss(reconstruction, well_hd_combined)

        loss_g = recons + kl_weight * kl + perceptual_weight * perceptual + hd_weight * hd
        loss_g.backward()
        optimizer_g.step()

        epoch_recons_loss += recons.item()
        epoch_kl_loss += kl.item()
        epoch_percep_loss += perceptual.item()
        epoch_hd_loss += hd.item()

        progress_bar.set_postfix({
            "recons": epoch_recons_loss / (step + 1),
            "kl": epoch_kl_loss / (step + 1),
            "percep": epoch_percep_loss / (step + 1),
            "hd": epoch_hd_loss / (step + 1),
        })

    epoch_recon_loss_list.append(epoch_recons_loss / len(train_loader))
    epoch_kl_loss_list.append(epoch_kl_loss / len(train_loader))
    epoch_percep_loss_list.append(epoch_percep_loss / len(train_loader))
    epoch_hd_loss_list.append(epoch_hd_loss / len(train_loader))

    # -------------------------- Validation -------------------------- #
    autoencoder.eval()
    val_recons_loss = 0.0
    val_kl_loss = 0.0
    val_percep_loss = 0.0
    val_hd_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)

            reconstruction, z_mu, z_sigma = autoencoder(images)

            kl = KL_loss(z_mu, z_sigma)
            recons = l2_loss(reconstruction.float(), images.float())
            perceptual = loss_perceptual(reconstruction.float(), images.float())
            hd = compute_hd_loss(reconstruction, well_hd_combined)

            val_recons_loss += recons.item()
            val_kl_loss += kl.item()
            val_percep_loss += perceptual.item()
            val_hd_loss += hd.item()

    val_len = len(val_loader)
    val_recon_loss_list.append(val_recons_loss / val_len)
    val_kl_loss_list.append(val_kl_loss / val_len)
    val_percep_loss_list.append(val_percep_loss / val_len)
    val_hd_loss_list.append(val_hd_loss / val_len)

    print(f"\n[Validation] Epoch {epoch}: "
          f"Recons: {val_recon_loss_list[-1]:.4f}, "
          f"KL: {val_kl_loss_list[-1]:.4f}, "
          f"Percep: {val_percep_loss_list[-1]:.4f}, "
          f"HD: {val_hd_loss_list[-1]:.4f}")
