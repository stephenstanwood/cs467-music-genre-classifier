# Description: Packages the per-piece mel-spectrograms into one file for CNN
#              training, using the song-level train/val/test split.

"""
Bundle the mel-spectrogram .npy files (one per 3-second piece) into a single
compressed .npz file, lined up with Dawson's split CSV so the CNN uses the same
train/val/test songs as the 57-feature model.

Usage (from the repo root):
    python scripts/preprocessing/package_mels.py

Output:
    Data/gtzan_mels.npz with these arrays, all in the same row order:
        X         (N, 128, 130) float16, log-mel spectrograms in dB
        label     (N,) genre name
        song_id   (N,) e.g. "blues.00000"
        split     (N,) "train", "val", or "test"
        filename  (N,) e.g. "blues.00000.4.wav"
    Data/gtzan_mels_preview.png, one example spectrogram per genre

Loading it for training:
    data = np.load("Data/gtzan_mels.npz")
    train = data["split"] == "train"
    X_train = data["X"][train][..., np.newaxis]   # add a channel axis for Conv2D
    y_train = data["label"][train]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import preprocess as pp

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SHAPE = (pp.N_MELS, 1 + pp.SEGMENT_SAMPLES // pp.HOP_LENGTH)


def load_mels(split_df, mels_dir):
    """Load each piece's spectrogram in the split CSV's row order.
    Rows whose .npy file is missing are dropped and reported."""
    arrays, keep, missing = [], [], []
    for i, filename in enumerate(split_df["filename"]):
        path = mels_dir / (Path(filename).stem + ".npy")
        if not path.is_file():
            missing.append(filename)
            continue
        mel = np.load(path)
        if mel.shape != EXPECTED_SHAPE:
            raise ValueError(f"{path.name} has shape {mel.shape}, expected {EXPECTED_SHAPE}. "
                             "Were the settings in preprocess.py changed?")
        arrays.append(mel)
        keep.append(i)
    return np.stack(arrays).astype(np.float16), split_df.iloc[keep], missing


def save_preview(X, labels, out_path):
    """Save one example spectrogram per genre, as a quick visual check."""
    genres = sorted(set(labels))
    cols = 5
    rows = -(-len(genres) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
    for ax, genre in zip(axes.flat, genres):
        idx = int(np.flatnonzero(labels == genre)[0])
        ax.imshow(X[idx].astype(np.float32), origin="lower", aspect="auto",
                  cmap="magma", vmin=-80, vmax=0)
        ax.set_title(genre)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(genres):]:
        ax.axis("off")
    fig.suptitle("One 3-second mel-spectrogram per genre (low frequencies at the bottom)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", default=ROOT / "Data/gtzan_57_features_split.csv")
    ap.add_argument("--mels-dir", default=ROOT / "scripts/preprocessing/features/mels")
    ap.add_argument("--out", default=ROOT / "Data/gtzan_mels.npz")
    args = ap.parse_args()

    split_df = pd.read_csv(args.split_csv, usecols=["filename", "song_id", "label", "split"])
    X, rows, missing = load_mels(split_df, Path(args.mels_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X,
        label=rows["label"].to_numpy(dtype=str),
        song_id=rows["song_id"].to_numpy(dtype=str),
        split=rows["split"].to_numpy(dtype=str),
        filename=rows["filename"].to_numpy(dtype=str),
    )
    preview = out.with_name(out.stem + "_preview.png")
    save_preview(X, rows["label"].to_numpy(dtype=str), preview)

    print(f"Saved {len(X)} spectrograms of shape {X.shape[1:]} to {out} "
          f"({out.stat().st_size / 1e6:.0f} MB)")
    print(rows["split"].value_counts().to_string())
    print(f"Preview image: {preview}")
    if missing:
        print(f"\n{len(missing)} piece(s) in the split CSV had no spectrogram file, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()