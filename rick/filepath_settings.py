""" File used to define file-path settings."""
from os import path
from pathlib import Path

# -------------------------------------------------------
# Project Root
# -------------------------------------------------------
ROOT_DIR = Path(path.dirname(path.abspath(__file__)))

# -------------------------------------------------------
# Data
# -------------------------------------------------------
DATA_DIR = ROOT_DIR / 'data'

GIF_METADATA_PATH = DATA_DIR / 'gif_metadata.p'
GIF_PROCESSED_METADATA_PATH = DATA_DIR / 'processed_metadata.p'

GIF_TRAIN_METADATA = DATA_DIR / 'train_metadata.p'
GIF_TRAIN_DATA_DIR = DATA_DIR / 'train_gif_data'

GIF_TEST_DATA_DIR = DATA_DIR / 'test_gif_data'
