import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.data import Dataset


# ------------------------ Dataset Setup ------------------------- #
def create_dataloader(data, batch_size, shuffle=False):
    '''
    Create a DataLoader from an array of geomodels.

    Parameters
    ----------
    data : ndarray
        The array of geomodels to load (shape is nr, nx, ny, nz).
    batch_size : int
        The number of samples per batch.
    shuffle : bool, optional
        Whether to shuffle the data at every epoch. Default is False.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader yielding batches of {"image": tensor} dicts.
    '''
    datalist = [{"image": torch.tensor(model)} for model in data]
    dataset  = Dataset(data=datalist)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ----------------------- Print Helper --------------------------- #
def print_losses(split, epoch, n_epochs, loss=None, recon=None, kl=None, percep=None, hd=None, gen=None, disc=None):
    '''
    Print a uniformly formatted loss summary for training or validation.
    Supports a single-loss mode (e.g. UNet diffusion) or a multi-loss mode (e.g. VAE).

    Parameters
    ----------
    split : str
        The dataset split label, e.g. "Train" or "Val".
    epoch : int
        The current epoch number.
    n_epochs : int
        The total number of epochs.
    loss : float, optional
        A single scalar loss (used in single-loss mode, e.g. UNet).
    recon : float, optional
        The reconstruction loss (VAE mode).
    kl : float, optional
        The KL divergence loss (VAE mode).
    percep : float, optional
        The perceptual loss (VAE mode).
    hd : float, optional
        The hard data loss (VAE mode).
    gen : float, optional
        The generator adversarial loss. Printed only if both gen and disc are provided.
    disc : float, optional
        The discriminator adversarial loss. Printed only if both gen and disc are provided.
    '''
    if loss is not None:
        print(f"[{split} @ Epoch {epoch}/{n_epochs}] Loss: {loss:.6f}")
        return
    adv_str = f" | Gen: {gen:.4f} | Disc: {disc:.4f}" if (gen is not None and disc is not None) else ""
    print(
        f"[{split} @ Epoch {epoch}/{n_epochs}] "
        f"Recon: {recon:.4f} | "
        f"KL: {kl:.4f} | "
        f"Percep: {percep:.4f} | "
        f"HD: {hd:.4f}"
        f"{adv_str}"
    )


# ------------------------ HD Loss Function ---------------------- #
def compute_hd_loss(y_pred, well_hd_all):
    '''
    Compute the hard data loss between the predicted geomodels and the conditioning points.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted geomodels (shape is batch, 1, nx, ny, nz).
    well_hd_all : ndarray
        The array of well locations and values (shape is nz*n_wells, 4, where 4 is [x, y, z, value]).

    Returns
    -------
    hd_loss : torch.Tensor
        The hard data loss scalar.
    '''
    ix = well_hd_all[:, 0].astype(int)
    iy = well_hd_all[:, 1].astype(int)
    iz = well_hd_all[:, 2].astype(int)
    v  = torch.from_numpy(well_hd_all[:, -1]).float().to(y_pred.device).repeat(y_pred.shape[0], 1)

    preds    = y_pred[:, 0, ix, iy, iz]
    hd_loss  = F.l1_loss(preds, v)  # or F.mse_loss

    return hd_loss


# -------------------- Hard Data Pickle Builder ------------------ #
def build_hard_data_pickle(models, well_loc):
    '''
    Build a dictionary of conditioning points (hard data) from the geomodels.

    Parameters
    ----------
    models : ndarray
        The multidimensional array of geomodels (shape is nr, nx, ny, nz).
    well_loc : dict
        The dictionary of well names to locations (x, y), e.g. {'I1': (11, 11), 'I2': (11, 54)}.

    Returns
    -------
    well_hd_all : dict
        Dictionary mapping well names to arrays of shape (nz, 4), where 4 is [x, y, z, value].
    '''
    models_T = models.transpose((0, 4, 3, 2, 1))
    nz       = models_T.shape[1]

    for wn, (ix, iy) in well_loc.items():
        ix -= 1
        iy -= 1
        for iz in range(nz):
            data = models_T[:, iz, iy, ix, 0]
            if data.max() != data.min():
                print(f'Not all models have the same value at location {ix, iy, iz}', wn)

    well_hd_all = {}
    for wn, (ix, iy) in well_loc.items():
        well_hd_all[wn] = []
        ix -= 1
        iy -= 1
        for iz in range(nz):
            well_hd_all[wn].append((ix, iy, iz, models_T[0, iz, iy, ix, 0]))
        well_hd_all[wn] = np.array(well_hd_all[wn])

    return well_hd_all


# -------------------- Hard Data Pickle I/O --------------------- #
def save_hard_data_pickle(hard_data, folder):
    '''
    Save the hard data dictionary to a pickle file.

    Parameters
    ----------
    hard_data : dict
        Dictionary mapping well names to arrays of shape (nz, 4), where 4 is [x, y, z, value].
    folder : str
        The folder path where the pickle file will be saved.
    '''
    with open(folder + '/well_hd.pickle', 'wb') as fid:
        pickle.dump(hard_data, fid)


def load_hard_data_pickle(folder):
    '''
    Load the hard data dictionary from a pickle file and concatenate all wells.

    Parameters
    ----------
    folder : str
        The folder path where the pickle file is stored.

    Returns
    -------
    well_hd_all : ndarray
        The concatenated array of all well locations and values (shape is nz*n_wells, 4,
        where 4 is [x, y, z, value]).
    '''
    with open(folder + 'well_hd.pickle', 'rb') as fid:
        well_hd = pickle.load(fid)

    for wn in well_hd:
        well_hd[wn][:, -1] = well_hd[wn][:, -1]

    well_hd_all = np.concatenate(list(well_hd.values()), axis=0)
    print('Total number of hard data:', well_hd_all.shape[0])

    return well_hd_all


# ----------------------- Geomodel Utils ------------------------- #
def model2tricat(model, thresh1, thresh2):
    '''
    Convert a continuous geomodel to a tri-categorical representation.

    Parameters
    ----------
    model : ndarray
        The continuous geomodel array (values in [0, 1]).
    thresh1 : float
        The lower threshold; values below are set to 0.
    thresh2 : float
        The upper threshold; values above are set to 1. Values in between are set to 0.5.

    Returns
    -------
    model_copy : ndarray
        The tri-categorical geomodel array.
    '''
    model_copy = np.copy(model)
    model_copy[model_copy < thresh1]                               = 0.
    model_copy[(model_copy >= thresh1) & (model_copy <= thresh2)]  = 0.5
    model_copy[model_copy > thresh2]                               = 1.

    return model_copy


# ------------------------- Generation --------------------------- #
def diffusion_generate_monai_large(inferer, unet, vae, scheduler, latent, n_steps, num_per_iter, nx, ny, nz, verbose=True):
    '''
    Generate an ensemble of geomodels using the latent diffusion inferer.

    Parameters
    ----------
    inferer : LatentDiffusionInferer
        The diffusion inferer initialised with the trained scheduler and scale factor.
    unet : DiffusionModelUNet
        The trained U-Net denoising model.
    vae : AutoencoderKL
        The trained variational autoencoder.
    scheduler : DDIMScheduler
        The noise scheduler to use during sampling.
    latent : torch.Tensor
        The input noise tensor (shape is nr, 1, nx_latent, ny_latent, nz_latent).
    n_steps : int
        The number of denoising steps in the diffusion process.
    num_per_iter : int
        The number of samples to generate per iteration to manage GPU memory.
    nx, ny, nz : int
        The spatial dimensions of the output geomodels in grid blocks.
    verbose : bool, optional
        Whether to show the diffusion progress bar. Default is True.

    Returns
    -------
    models : ndarray
        Generated geomodels (shape is nr, nx, ny, nz).
    '''
    num_gen  = latent.shape[0]
    num_iter = max(1, num_gen // num_per_iter)
    device   = latent.device

    synthetic_images = torch.zeros((num_gen, 1, nx, ny, nz), device=device)

    with torch.no_grad():
        for i in range(num_iter):
            noise = latent[i * num_per_iter : (i + 1) * num_per_iter]
            scheduler.set_timesteps(num_inference_steps=n_steps)

            synthetic_images_single = inferer.sample(
                input_noise=noise,
                autoencoder_model=vae,
                diffusion_model=unet,
                scheduler=scheduler,
                verbose=verbose,
            )
            synthetic_images[i * num_per_iter : (i + 1) * num_per_iter] = synthetic_images_single
            del synthetic_images_single

    return synthetic_images.detach().cpu().numpy()[:, 0]


def compute_hd_accuracy(models, well_hd_all):
    '''
    Compute the hard data accuracy of generated geomodels.

    Parameters
    ----------
    models : ndarray
        The generated geomodels (shape is nr, nx, ny, nz).
    well_hd_all : ndarray
        The array of well locations and values (shape is nz*n_wells, 4, where 4 is [x, y, z, value]).

    Returns
    -------
    str
        A summary string reporting the percentage of correct hard data points and total errors.
    '''
    errors       = 0
    total_points = models.shape[0] * well_hd_all.shape[0]

    for i in range(models.shape[0]):
        for loc in well_hd_all:
            if models[i, int(loc[0]), int(loc[1]), int(loc[2])] != loc[3]:
                errors += 1

    return (
        f"Hard data is correct {100 - errors / total_points * 100:.2f}% of the time "
        f"with a total of {errors} errors out of {total_points} points"
    )



def KL_loss(z_mu, z_sigma):
    '''
    Compute the KL divergence loss for a VAE latent space.

    Parameters
    ----------
    z_mu : torch.Tensor
        The mean of the latent distribution (shape is batch, latent_channels, *spatial_dims).
    z_sigma : torch.Tensor
        The standard deviation of the latent distribution (same shape as z_mu).

    Returns
    -------
    torch.Tensor
        The mean KL divergence loss scalar across the batch.
    '''
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=[1, 2, 3, 4]
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]
