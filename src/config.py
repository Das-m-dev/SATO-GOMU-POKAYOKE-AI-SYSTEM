from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#source images: use data_aug for the smoke est now, data_roi later
SOURCE_DIR = ROOT / "data_aug" # change later to ROOT / "data_roi"
DATASET_DIR = ROOT / "datasets" / "seal" # built by make_split.py
OUTPUT_DIR = ROOT / "outputs" / "anomalib"

IMAGE_SIZE = (384, 512) # (height, width) fed to the model
SEED = 42