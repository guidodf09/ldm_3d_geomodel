'''
File: train_unet.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to train a U-net to learn the de-noising process in the latent space of latent diffusion models
'''

# ---------------------------- Imports ---------------------------- #
import os
import h5py
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from monai.data import Dataset

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler

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
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

n_epochs = 1000
batch_size = 2
val_interval = 10
save_interval = 100
lr = 1e-4

# -------------------------- Load Data --------------------------- #
with h5py.File(h5_file_path, 'r') as f:
    models_loaded = model2tricat(np.array(f["data"])[:] / 255., 0.25, 0.8)

# Reproducible shuffling
indices = np.arange(models_loaded.shape[0])
np.random.shuffle(indices)
models_loaded = models_loaded[indices]

nx, ny, nz = models_loaded.shape[2:]

# Well locations for conditioning
well_loc = {
    'i1': (16, 16), 'i2': (16, 64), 'i3': (16, 112), 'i4': (64, 16),
    'p1': (64, 64), 'p2': (64, 112), 'p3': (112, 16),
    'p4': (112, 64), 'p5': (112, 112)
}

# Hard data setup
hard_data_dict = build_hard_data_pickle(models_loaded, well_loc)
save_hard_data_pickle(hard_data_dict, case_dir)
well_hd_combined = np.concatenate(list(load_hard_data_pickle(case_dir).values()), axis=0)

# ------------------------ Dataset Setup ------------------------- #
def create_dataloader(data, batch_size, shuffle=True):
    datalist = [{"image": torch.tensor(model)} for model in data]
    dataset = Dataset(data=datalist)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_dataloader(models_loaded[:2400], batch_size)
val_loader   = create_dataloader(models_loaded[2400:2700], batch_size, shuffle=False)
test_loader  = create_dataloader(models_loaded[2700:], batch_size, shuffle=False)

# ------------------------ Load Pretrained VAE -------------------- #
vae_epoch = 1000
trained_vae_path = os.path.join(case_dir, f"trained_vae_{vae_epoch}.pt")
vae_pickle_path  = os.path.join(case_dir, "autoencoder_properties.pkl")

with open(vae_pickle_path, "rb") as pkl_file:
    vae_properties = pickle.load(pkl_file)

autoencoder = AutoencoderKL(**vae_properties).to(device)
autoencoder.load_state_dict(torch.load(trained_vae_path, map_location=device))
autoencoder.eval()

# ------------------------ UNet Setup ----------------------------- #
unet_properties = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 1,
    "num_channels": (64, 128, 256),
    "num_res_blocks": 1,
    "attention_levels": (False, True, True),
    "num_head_channels": (0, 128, 256),
}

unet = DiffusionModelUNet(**unet_properties).to(device)
optimizer_diff = torch.optim.Adam(unet.parameters(), lr=lr)
scaler = GradScaler()

# Save model config
unet_pickle_path = os.path.join(case_dir, "unet_properties.pkl")
with open(unet_pickle_path, "wb") as f:
    pickle.dump(unet_properties, f)

# ------------------------ Diffusion Setup ------------------------ #
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="scaled_linear_beta",
    beta_start=0.0015,
    beta_end=0.0195
)
inferer = LatentDiffusionInferer(scheduler, scale_factor=1.0)

trained_unet_path = os.path.join(case_dir, "trained_unet")

# --------------------------- Training ---------------------------- #
epoch_loss_list = []
val_loss_list = []

for epoch in range(n_epochs):
    print(f"\nEpoch {epoch + 1}/{n_epochs}")

    # ------------------- Training ------------------- #
    unet.train()
    train_loss = 0.0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
    progress_bar.set_description("Training")

    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer_diff.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            noise = torch.randn_like(images)
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
        progress_bar.set_postfix({"Train Loss": f"{train_loss / (step + 1):.6f}"})

    epoch_loss = train_loss / len(train_loader)
    epoch_loss_list.append(epoch_loss)

    # ------------------- Validation ------------------- #
    if (epoch + 1) % val_interval == 0:
        unet.eval()
        val_loss = 0.0

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                with autocast(enabled=True):
                    noise = torch.randn_like(images)
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
        print(f"[Validation] Epoch {epoch + 1}: Loss = {val_loss:.6f}")

    # ------------------- Save Model ------------------- #
    if (epoch + 1) % save_interval == 0:
        torch.save(unet.state_dict(), f"{trained_unet_path}_{epoch + 1}.pt")
        print(f"Model saved at epoch {epoch + 1}")

# --------------------------- Test Evaluation --------------------------- #
unet.eval()
test_loss = 0.0

with torch.no_grad():
    for step, batch in enumerate(test_loader):
        images = batch["image"].to(device)
        with autocast(enabled=True):
            noise = torch.randn_like(images)
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

            test_loss += F.mse_loss(noise_pred.float(), noise.float()).item()

test_loss /= len(test_loader)
print(f"\n[Test] Final Loss: {test_loss:.6f}")

print("\nTraining complete")
