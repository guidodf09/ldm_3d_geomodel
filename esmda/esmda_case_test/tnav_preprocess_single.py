# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import yaml
import numpy as np
from utils import create_multiple_run_folders

# ─── Config ───────────────────────────────────────────────────────────────────
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# ─── Directories ──────────────────────────────────────────────────────────────
project_base    = os.getcwd() + '/'
template_folder = os.path.join(project_base, 'template_folder')
true_model_path = os.path.join(project_base, cfg['paths']['true_model'])

single_folder = os.environ.get('SINGLE_RUN_PATH')
if not single_folder:
    raise EnvironmentError("ERROR: SINGLE_RUN_PATH environment variable is not set")
os.makedirs(single_folder, exist_ok=True)

# ─── Load & Transform ─────────────────────────────────────────────────────────
m_single = np.load(true_model_path)
m_single = np.flip(m_single.transpose(0, 3, 2, 1), axis=1)

# ─── Create Run Folder ────────────────────────────────────────────────────────
print(f"Creating single run folder in {single_folder}...")
create_multiple_run_folders(template_folder, m_single, single_folder)
print("Done.")