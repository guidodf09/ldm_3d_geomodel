'''
File: train_vae.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to train a variational autoencoder (VAE) to learn the mapping between geomodel space and low-dimensional latent space for latent diffusion models
'''

# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import h5py
import pickle

import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import L1Loss

from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from generative.losses import PerceptualLoss, PatchAdversarialLoss

from utils import (
    KL_loss,
    model2tricat,
    build_hard_data_pickle,
    save_hard_data_pickle,
    create_dataloader,
    compute_hd_loss,
    print_losses
)

# ─── Config ───────────────────────────────────────────────────────────────────
with open('config_vae.yaml') as f:
    cfg = yaml.safe_load(f)

# ─── Directories ──────────────────────────────────────────────────────────────
case_dir          = cfg['paths']['case_dir']
h5_file_path      = cfg['paths']['h5_file']
vae_dir          = os.path.join(case_dir, cfg['paths']['vae_dir'])
os.makedirs(vae_dir, exist_ok=True)
trained_vae_path = os.path.join(vae_dir, 'trained_vae')
vae_pickle_path  = os.path.join(vae_dir, 'autoencoder_properties.pkl')
vae_txt_path     = os.path.join(vae_dir, 'autoencoder_properties.txt')
os.makedirs(case_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── Training Settings ────────────────────────────────────────────────────────
n_epochs                     = cfg['training']['n_epochs']
batch_size                   = cfg['training']['batch_size']
val_interval                 = cfg['training']['val_interval']
save_interval                = cfg['training']['save_interval']
autoencoder_warm_up_n_epochs = cfg['training']['warm_up_epochs']
multi_gpu_vae                = cfg['training']['multi_gpu']
seed                         = cfg['training']['seed']

# ─── Loss Weights ─────────────────────────────────────────────────────────────
kl_weight         = cfg['loss_weights']['kl']
perceptual_weight = cfg['loss_weights']['perceptual']
hd_weight         = cfg['loss_weights']['hd']
adv_weight        = cfg['loss_weights']['adv']

# ─── Data Split ───────────────────────────────────────────────────────────────
train_end = cfg['data_split']['train_end']
val_start = cfg['data_split']['val_start']
val_end   = cfg['data_split']['val_end']

# ─── Load Data ────────────────────────────────────────────────────────────────
with h5py.File(h5_file_path, 'r') as f:
    models_loaded = model2tricat(np.array(f["data"])[:] / 255.,
                                 cfg['facies']['thresh1'],
                                 cfg['facies']['thresh2'])

np.random.seed(seed)
np.random.shuffle(models_loaded)

# ─── Hard Data ────────────────────────────────────────────────────────────────
well_loc = {name: tuple(coords) for name, coords in cfg['wells'].items()}

hard_data_dict   = build_hard_data_pickle(models_loaded, well_loc)
save_hard_data_pickle(hard_data_dict, case_dir)
well_hd_combined = np.concatenate(list(hard_data_dict.values()), axis=0)

# ─── Dataloaders ──────────────────────────────────────────────────────────────
train_loader = create_dataloader(models_loaded[:train_end],        batch_size, shuffle=True)
val_loader   = create_dataloader(models_loaded[val_start:val_end], batch_size)
test_loader  = create_dataloader(models_loaded[val_end:],          batch_size)

# ─── VAE ──────────────────────────────────────────────────────────────────────
vae_properties = {
    'spatial_dims':     cfg['vae']['spatial_dims'],
    'in_channels':      cfg['vae']['in_channels'],
    'out_channels':     cfg['vae']['out_channels'],
    'num_channels':     tuple(cfg['vae']['num_channels']),
    'latent_channels':  cfg['vae']['latent_channels'],
    'num_res_blocks':   cfg['vae']['num_res_blocks'],
    'norm_num_groups':  cfg['vae']['norm_num_groups'],
    'attention_levels': tuple(cfg['vae']['attention_levels']),
}

with open(vae_pickle_path, 'wb') as f:
    pickle.dump(vae_properties, f)

with open(vae_txt_path, 'w') as f:
    f.write("\n".join(f"{k}: {v}" for k, v in vae_properties.items()))

autoencoder = AutoencoderKL(**vae_properties).to(device)
if multi_gpu_vae:
    autoencoder = nn.DataParallel(autoencoder)

# ─── Discriminator ────────────────────────────────────────────────────────────
discriminator = PatchDiscriminator(
    spatial_dims=cfg['discriminator']['spatial_dims'],
    num_layers_d=cfg['discriminator']['num_layers_d'],
    num_channels=cfg['discriminator']['num_channels'],
    in_channels=cfg['discriminator']['in_channels'],
    out_channels=cfg['discriminator']['out_channels'],
).to(device)

# ─── Optimizers & Losses ──────────────────────────────────────────────────────
optimizer_g = torch.optim.Adam(autoencoder.parameters(),   lr=cfg['training']['lr'])
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg['training']['lr'])

l1_loss        = L1Loss()
percep_loss_fn = PerceptualLoss(
    spatial_dims=3, is_fake_3d=True, fake_3d_ratio=0.2,
    network_type=cfg['training']['perceptual_network']
).to(device)
adv_loss = PatchAdversarialLoss(criterion='least_squares')

# ─── Loss Tracking ────────────────────────────────────────────────────────────
epoch_recon_loss_list = []
epoch_gen_loss_list   = []
epoch_disc_loss_list  = []
epoch_hd_loss_list    = []

# ─── Training Loop ────────────────────────────────────────────────────────────
for epoch in range(n_epochs):

    autoencoder.train()
    discriminator.train()

    epoch_recon, epoch_kl, epoch_percep, epoch_hd, epoch_gen, epoch_disc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    train_loop.set_description(f"Epoch {epoch + 1}/{n_epochs}")

    for step, batch in train_loop:
        images = batch["image"].to(device)

        # -------- Generator --------
        optimizer_g.zero_grad(set_to_none=True)

        reconstruction, z_mu, z_sigma = autoencoder(images)

        kl         = KL_loss(z_mu, z_sigma)
        recon_loss = l1_loss(reconstruction.float(), images.float())
        perceptual = percep_loss_fn(reconstruction.float(), images.float())
        hd         = compute_hd_loss(reconstruction, well_hd_combined)

        loss_g = recon_loss + kl_weight * kl + perceptual_weight * perceptual + hd_weight * hd

        if epoch > autoencoder_warm_up_n_epochs:
            logits_fake    = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g        += adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        # -------- Discriminator --------
        if epoch > autoencoder_warm_up_n_epochs:
            optimizer_d.zero_grad(set_to_none=True)

            logits_fake  = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake  = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

            logits_real  = discriminator(images.contiguous().detach())[-1]
            loss_d_real  = adv_loss(logits_real, target_is_real=True,  for_discriminator=True)

            discriminator_loss = 0.5 * (loss_d_fake + loss_d_real)
            loss_d = adv_weight * discriminator_loss
            loss_d.backward()
            optimizer_d.step()

        # -------- Accumulate --------
        epoch_recon  += recon_loss.item()
        epoch_kl     += kl.item()
        epoch_percep += perceptual.item()
        epoch_hd     += hd.item()
        if epoch > autoencoder_warm_up_n_epochs:
            epoch_gen  += generator_loss.item()
            epoch_disc += discriminator_loss.item()

        train_loop.set_postfix({
            "recon":  f"{epoch_recon  / (step + 1):.4f}",
            "kl":     f"{epoch_kl     / (step + 1):.4f}",
            "percep": f"{epoch_percep / (step + 1):.4f}",
            "hd":     f"{epoch_hd     / (step + 1):.4f}",
            "gen":    f"{epoch_gen    / (step + 1):.4f}",
            "disc":   f"{epoch_disc   / (step + 1):.4f}",
        })

    n_train    = step + 1
    adv_active = epoch > autoencoder_warm_up_n_epochs

    epoch_recon_loss_list.append(epoch_recon  / n_train)
    epoch_hd_loss_list.append(epoch_hd        / n_train)
    epoch_gen_loss_list.append(epoch_gen       / n_train if adv_active else 0.0)
    epoch_disc_loss_list.append(epoch_disc     / n_train if adv_active else 0.0)

    print_losses(
        split="Train", epoch=epoch + 1, n_epochs=n_epochs,
        recon=epoch_recon   / n_train,
        kl=epoch_kl         / n_train,
        percep=epoch_percep / n_train,
        hd=epoch_hd         / n_train,
        gen=epoch_gen       / n_train if adv_active else None,
        disc=epoch_disc     / n_train if adv_active else None,
    )

    # ─── Validation ───────────────────────────────────────────────────────────
    if (epoch + 1) % val_interval == 0:
        autoencoder.eval()
        val_recon, val_kl, val_percep, val_hd = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)

                reconstruction, z_mu, z_sigma = autoencoder(images)
                kl         = KL_loss(z_mu, z_sigma)
                recon_loss = l1_loss(reconstruction.float(), images.float())
                perceptual = percep_loss_fn(reconstruction.float(), images.float())
                hd         = compute_hd_loss(reconstruction, well_hd_combined)

                val_recon  += recon_loss.item()
                val_kl     += kl.item()
                val_percep += perceptual.item()
                val_hd     += hd.item()

        n_val = len(val_loader)
        print_losses(
            split="Val", epoch=epoch + 1, n_epochs=n_epochs,
            recon=val_recon   / n_val,
            kl=val_kl         / n_val,
            percep=val_percep / n_val,
            hd=val_hd         / n_val,
        )

    # ─── Checkpoint ───────────────────────────────────────────────────────────
    if (epoch + 1) % save_interval == 0:
        torch.save(autoencoder.state_dict(), f"{trained_vae_path}{epoch + 1}.pt")
        print(f"Model saved to {trained_vae_path}{epoch + 1}.pt")

# ─── Save Final Model ─────────────────────────────────────────────────────────
torch.save(autoencoder.state_dict(), f"{trained_vae_path}.pt")
print(f"Final model saved to {trained_vae_path}.pt")

# ─── Compute and Save Scale Factor ───────────────────────────────────────────
with torch.no_grad():
    sample       = next(iter(train_loader))
    z            = autoencoder.module.encode_stage_2_inputs(sample["image"].to(device))
    scale_factor = (1 / torch.std(z)).item()
print(f"Scaling factor set to {scale_factor:.4f}")

with open('config_unet.yaml') as f:
    cfg_unet = yaml.safe_load(f)

cfg_unet['vae']['scale_factor'] = scale_factor

with open('config_unet.yaml', 'w') as f:
    yaml.dump(cfg_unet, f, default_flow_style=False)

print(f"Scale factor written to config_unet.yaml")
