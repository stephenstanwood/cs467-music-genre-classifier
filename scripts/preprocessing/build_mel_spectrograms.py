"""
package_mels.py — bundle the per-piece mel-spectrogram .npy files into one
.npz file for CNN training, lined up with Dawson's train/val/test split.

Usage (from the repo root):
    python scripts/preprocessing/package_mels.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import preprocess as pp

ROOT = Path(__file__).resolve().parents[2]     # repo root
EXPECTED_SHAPE = (pp.N_MELS, 130)              # 128 mel bands x 130 time frames


def load_mels(split_df, mels_dir):
    """Load each piece's spectrogram, in the same row order as the split CSV.
    Skip and record any piece whose .npy file is missing.
    Raise an error if a spectrogram isn't EXPECTED_SHAPE."""
    # TODO: for each filename (blues.00000.3.wav -> blues.00000.3.npy):
    #       np.load it, check its shape, collect it
    # TODO: return stacked array (as float16), the matching rows, missing list
    pass


def save_preview(X, labels, out_path):
    """Save one example spectrogram per genre as a PNG, as a visual check."""
    # TODO: grid of subplots, one per genre, plt.imshow each, savefig
    pass


def main():
    # 1. Read command-line options (with defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", default=ROOT / "Data/gtzan_57_features_split.csv")
    parser.add_argument("--mels-dir", default=ROOT / "scripts/preprocessing/features/mels")
    parser.add_argument("--out", default=ROOT / "Data/gtzan_mels.npz")
    args = parser.parse_args()

    # 2. Load Dawson's split CSV (filename, song_id, label, split)
    # TODO: pd.read_csv(...)

    # 3. Load the matching spectrograms
    # TODO: X, rows, missing = load_mels(...)

    # 4. Save everything into one compressed file
    # TODO: np.savez_compressed(out, X=..., label=..., song_id=..., split=..., filename=...)

    # 5. Save the preview image and print a summary
    # TODO: save_preview(...), print counts per split and any missing pieces


if __name__ == "__main__":
    main()