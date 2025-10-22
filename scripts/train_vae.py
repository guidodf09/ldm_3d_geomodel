# ---------------------------- Imports ---------------------------- #
import os
import h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from tqdm import tqdm

from monai.data import Dataset
from generative.networks.nets import AutoencoderKL
from generative.losses import PerceptualLoss

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training settings
n_epochs = 1000
batch_size = 2
val_interval = 10
save_interval = 50

# Loss Weights
kl_weight = 1e-6
perceptual_weight = 0.001
hd_weight = 1e-2

trained_vae_path = os.path.join(case_dir, 'trained_vae')

# -------------------------- Load Data --------------------------- #
with h5py.File(h5_file_path, 'r') as f:
    models_loaded = model2tricat(np.array(f["data"])[:] / 255., 0.25, 0.8)

seed = 0
np.random.seed(seed)
np.random.shuffle(models_loaded)
size = models_loaded.shape[2:]

# Well locations for hard data
well_loc = {
    'i1': (16, 16), 'i2': (16, 64), 'i3': (16, 112), 'i4': (64, 16),
    'p1': (64, 64), 'p2': (64, 112), 'p3': (112, 16), 'p4': (112, 64), 'p5': (112, 112)
}

# Save and load hard data
hard_data_dict = build_hard_data_pickle(models_loaded, well_loc)
save_hard_data_pickle(hard_data_dict, case_dir)
well_hd_combined = np.concatenate(list(load_hard_data_pickle(case_dir).values()), axis=0)

# ------------------------ Dataset Setup ------------------------- #
def create_dataloader(data, batch_size):
    datalist = [{"image": torch.tensor(model)} for model in data]
    dataset = Dataset(data=datalist)
    return DataLoader(dataset, batch_size=batch_size)

# Data split: 2400 train, 300 val, 300 test
train_loader = create_dataloader(models_loaded[:2400], batch_size)
val_loader   = create_dataloader(models_loaded[2400:2700], batch_size)
test_loader  = create_dataloader(models_loaded[2700:], batch_size)

# ------------------------ Model Setup --------------------------- #
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
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
l2_loss = MSELoss()
percep_loss_fn = PerceptualLoss(
    spatial_dims=3, is_fake_3d=True, fake_3d_ratio=0.2
).to(device)

# ------------------------ HD Loss Function ---------------------- #
def compute_hd_loss(y_pred, hd_points):
    ix, iy, iz = hd_points[:, 0].astype(int), hd_points[:, 1].astype(int), hd_points[:, 2].astype(int)
    v = torch.from_numpy(hd_points[:, -1]).float().to(device).repeat(y_pred.shape[0], 1)
    preds = y_pred[:, 0, ix, iy, iz]
    return F.mse_loss(preds, v)

# --------------------------- Training Loop --------------------------- #
for epoch in range(n_epochs):
    print(f"\nEpoch {epoch + 1}/{n_epochs}")

    if (epoch + 1) % save_interval == 0:
        torch.save(autoencoder.state_dict(), f"{trained_vae_path}_{epoch + 1}.pt")

    autoencoder.train()
    train_recon, train_kl, train_percep, train_hd = 0.0, 0.0, 0.0, 0.0

    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    train_loop.set_description("Training")

    for step, batch in train_loop:
        images = batch["image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        recon, z_mu, z_sigma = autoencoder(images)

        kl = KL_loss(z_mu, z_sigma)
        recon_loss = l2_loss(recon, images)
        perceptual = percep_loss_fn(recon, images)
        hd = compute_hd_loss(recon, well_hd_combined)

        loss = recon_loss + kl_weight * kl + perceptual_weight * perceptual + hd_weight * hd
        loss.backward()
        optimizer.step()

        train_recon += recon_loss.item()
        train_kl += kl.item()
        train_percep += perceptual.item()
        train_hd += hd.item()

        train_loop.set_postfix({
            "Recon": f"{train_recon / (step + 1):.4f}",
            "KL": f"{train_kl / (step + 1):.4f}",
            "Percep": f"{train_percep / (step + 1):.4f}",
            "HD": f"{train_hd / (step + 1):.4f}"
        })

    # ------------------------ Validation  ------------------------- #
    if (epoch + 1) % val_interval == 0:
        autoencoder.eval()
        val_recon, val_kl, val_percep, val_hd = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)

                recon, z_mu, z_sigma = autoencoder(images)
                kl = KL_loss(z_mu, z_sigma)
                recon_loss = l2_loss(recon, images)
                perceptual = percep_loss_fn(recon, images)
                hd = compute_hd_loss(recon, well_hd_combined)

                val_recon += recon_loss.item()
                val_kl += kl.item()
                val_percep += perceptual.item()
                val_hd += hd.item()

        n_val = len(val_loader)
        print(f"[Validation @ Epoch {epoch+1}] Recon:
