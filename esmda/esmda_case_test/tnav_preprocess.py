# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import numpy as np
from utils import create_multiple_run_folders

# ─── Directories ──────────────────────────────────────────────────────────────
project_base    = os.getcwd() + '/'
template_folder = os.path.join(project_base, 'template_folder')
results_folder  = os.path.join(project_base, 'results/')

runs_folder = os.environ.get('RUN_PATH')
if not runs_folder:
    raise EnvironmentError("ERROR: RUN_PATH environment variable is not set")
os.makedirs(runs_folder, exist_ok=True)

# ─── Load & Transform ─────────────────────────────────────────────────────────
m_cond = np.load(os.path.join(results_folder, 'current.npy'))
m_cond = np.flip(m_cond.transpose(0, 3, 2, 1), axis=1)

# ─── Create Run Folders ───────────────────────────────────────────────────────
print(f"Creating {len(m_cond)} run folders in {runs_folder}...")
create_multiple_run_folders(template_folder, m_cond, runs_folder)
print("Done.")