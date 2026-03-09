# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import shutil
import argparse
import yaml
import numpy as np
from utils import extract_sim_output, files_from_folder, generate_all_labels

# ─── Config ───────────────────────────────────────────────────────────────────
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

N_prod           = cfg['simulation']['N_prod']
N_inj            = cfg['simulation']['N_inj']
Nt               = cfg['simulation']['Nt']
results_filename = cfg['simulation']['results_file']

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Postprocess ensemble simulation output.")
parser.add_argument('--index', type=int, required=True, help='Ensemble iteration index')
args = parser.parse_args()
i_na = args.index

# ─── Directories ──────────────────────────────────────────────────────────────
main_dir    = os.path.join(os.getcwd(), 'results/')
results_dir = os.environ.get('RUN_PATH')
if not results_dir:
    raise EnvironmentError("ERROR: RUN_PATH environment variable is not set")

# ─── Extract & Save ───────────────────────────────────────────────────────────
print(f"Postprocessing ensemble iteration {i_na} from {results_dir}...")

sorted_full_paths  = files_from_folder(results_dir, results_filename)
labels             = generate_all_labels(N_prod, N_inj)
all_data_all_times = extract_sim_output(sorted_full_paths, Nt, labels)

out_path = os.path.join(main_dir, f'all_data_all_times_iter{i_na}.npy')
np.save(out_path, all_data_all_times)
print(f"Saved output to {out_path}")

# ─── Cleanup ──────────────────────────────────────────────────────────────────
print(f"Removing run folder {results_dir}...")
shutil.rmtree(results_dir)
print("Done.")