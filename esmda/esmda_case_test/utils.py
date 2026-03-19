import os
import shutil
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

_perm_raw = cfg['facies']['perm']
_poro_raw = cfg['facies']['poro']
PERM_MAP  = {float(k): v for k, v in _perm_raw.items()}
PORO_MAP  = {float(k): v for k, v in _poro_raw.items()}


# ─── Folder / File Utilities ──────────────────────────────────────────────────

def create_multiple_run_folders(template_folder, all_facies, base_folder):
    """
    Create run folders from a template, writing permeability and porosity
    files derived from the given facies array.

    Parameters
    ----------
    template_folder : str
        Path to the template folder.
    all_facies : ndarray, shape (Nr, Nx, Ny, Nz)
        Facies values (0, 1, or 2) for each realisation.
    base_folder : str
        Base directory where run_0, run_1, ... folders will be created.
    """
    for i, facies in enumerate(all_facies):
        folder_path = os.path.join(base_folder, f"run_{i}")
        shutil.copytree(template_folder, folder_path, dirs_exist_ok=True)

        perm = np.vectorize(PERM_MAP.get)(np.floor(facies))
        poro = np.vectorize(PORO_MAP.get)(facies)

        flat_perm = perm.flatten()
        for fname, header in [('PERMX.INC', 'PERMX'), ('PERMY.INC', 'PERMY'), ('PERMZ.INC', 'PERMZ')]:
            np.savetxt(os.path.join(folder_path, fname), flat_perm,
                       fmt='%.1f', header=header, footer='/', comments='')

        np.savetxt(os.path.join(folder_path, 'PORO.INC'), poro.flatten(),
                   fmt='%.4f', header='PORO', footer='/', comments='')


def files_from_folder(folder, results_file):
    """
    List files with the same name across all numerically-sorted subfolders.

    Parameters
    ----------
    folder : str
        Parent folder containing run_0, run_1, ... subfolders.
    results_file : str
        Relative path to the result file within each subfolder.

    Returns
    -------
    list of str
    """
    entries    = sorted(os.listdir(folder), key=lambda x: int(x.split('_')[-1]))
    full_paths = [os.path.join(folder, e) for e in entries]
    return [p + results_file for p in full_paths]


# ─── Label Generation ─────────────────────────────────────────────────────────

def generate_all_labels(N_prod, N_inj):
    """
    Generate producer and injector label strings.

    Parameters
    ----------
    N_prod : int
    N_inj  : int

    Returns
    -------
    list of str
        WWPR_PROD*, WOPR_PROD*, WWIR_INJ* labels concatenated.
    """
    return (
        [f"WWPR_PROD{i}" for i in range(1, N_prod + 1)] +
        [f"WOPR_PROD{i}" for i in range(1, N_prod + 1)] +
        [f"WWIR_INJ{i}"  for i in range(1, N_inj  + 1)]
    )


# ─── Simulation Output Processing ─────────────────────────────────────────────

def process_file(file_path, labels, skip_lines=10, read_lines=30, dt=50, Nt=30, cols_to_rm=5):
    """
    Parse a tNavigator RSM file into a (Nt, N_QoI+1) numpy array.

    Parameters
    ----------
    file_path  : str
    labels     : list of str
    skip_lines : int
    read_lines : int
    dt         : int    Time step size (days)
    Nt         : int    Number of time steps
    cols_to_rm : int    Leading columns to drop

    Returns
    -------
    ndarray, shape (Nt, N_QoI + 1)
    """
    maxt      = dt * Nt
    file_path = Path(file_path)
    tmp_path  = file_path.parent / '_tmp_processed.txt'

    with open(file_path, 'r') as f:
        lines = f.readlines()

    filtered, i = [], 0
    while i < len(lines):
        i += skip_lines
        filtered.extend(lines[i:i + read_lines])
        i += read_lines

    tmp_path.write_text(''.join(l.strip() + '\n' for l in filtered))

    try:
        df          = pd.read_csv(tmp_path, delim_whitespace=True, header=None)
        max_chunks  = df.shape[0] // Nt
        chunks      = [df.iloc[j*Nt:(j+1)*Nt].reset_index(drop=True) for j in range(max_chunks)]
        df_combined = pd.concat(chunks, axis=1)

        df_filtered = df_combined.dropna(axis=1).iloc[:, cols_to_rm:]
        df_final    = df_filtered.loc[:, ~df_filtered.isin([maxt, dt]).any()]
        df_final.insert(0, "DAY", np.linspace(dt, maxt, Nt))
    finally:
        tmp_path.unlink(missing_ok=True)

    return np.array(df_final)


def extract_sim_output(sorted_full_paths, Nt, labels):
    """
    Extract simulation output from multiple RSM files.

    Parameters
    ----------
    sorted_full_paths : list of str
    Nt     : int    Number of time steps
    labels : list of str

    Returns
    -------
    ndarray, shape (Nr, Nt, N_QoI)
    """
    Nr    = len(sorted_full_paths)
    N_QoI = len(labels)
    out   = np.zeros((Nr, Nt, N_QoI + 1))

    for i, path in enumerate(sorted_full_paths):
        out[i] = process_file(path, labels)

    return out[:, :, 1:]


# ─── Facies / Model Utilities ─────────────────────────────────────────────────

def model2tricat(model, thresh1, thresh2):
    """Binarise continuous model values into three facies categories (0, 1, 2)."""
    out = np.zeros_like(model)
    out[(model >= thresh1) & (model <= thresh2)] = 1.
    out[model > thresh2] = 2.
    return out


# ─── Latent Space Utilities ───────────────────────────────────────────────────

def csi_to_z(csi, latent_x, latent_y, latent_z):
    """
    Unflatten ES-MDA parameter matrix back to latent tensor.

    Parameters
    ----------
    csi : ndarray, shape (nx*ny*nz, nr)

    Returns
    -------
    z : ndarray, shape (nr, latent_x, latent_y, latent_z)
    """
    nr = csi.shape[1]
    return np.stack([csi[:, i].reshape(latent_x, latent_y, latent_z) for i in range(nr)])
    
    
def z_to_csi(z):
    """
    Flatten latent tensor to ES-MDA parameter matrix.

    Parameters
    ----------
    z : ndarray, shape (nr, nx_latent, ny_latent, nz_latent)

    Returns
    -------
    csi : ndarray, shape (nx_latent*ny_latent*nz_latent, nr)
    """
    nr = z.shape[0]
    return np.stack([z[i].flatten() for i in range(nr)], axis=1)


# ─── Diffusion Generation ─────────────────────────────────────────────────────

def diffusion_generate_monai_large(inferer, unet, vae, scheduler, n_steps, latent,
                                    num_per_iter, nx, ny, nz):
    """
    Generate geomodels from latents using the LDM inferer in batches.

    Parameters
    ----------
    inferer      : LatentDiffusionInferer
    unet         : DiffusionModelUNet
    vae          : AutoencoderKL
    scheduler    : DDIMScheduler
    n_steps      : int
    latent       : Tensor, shape (nr, 1, nx_l, ny_l, nz_l)
    num_per_iter : int    Batch size for generation
    nx, ny, nz   : int    Output spatial dimensions

    Returns
    -------
    models : ndarray, shape (nr, nx, ny, nz)
    """
    nr     = latent.shape[0]
    n_iter = max(1, nr // num_per_iter)
    out    = torch.zeros((nr, 1, nx, ny, nz), device='cuda')

    for i in range(n_iter):
        lo, hi = i * num_per_iter, (i + 1) * num_per_iter
        scheduler.set_timesteps(num_inference_steps=n_steps)
        batch = inferer.sample(
            input_noise=latent[lo:hi],
            autoencoder_model=vae,
            diffusion_model=unet,
            scheduler=scheduler
        )
        out[lo:hi] = batch
        del batch

    return out.detach().cpu().numpy()[:, 0]
