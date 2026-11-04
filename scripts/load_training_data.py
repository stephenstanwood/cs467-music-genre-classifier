# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Loads cached Mel spectrograms for training and validation.

import argparse
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = ("X", "label", "song_id", "split", "filename")
EXPECTED_SHAPE = (128, 130)


def check_cache(data):
    """Check the cache format and song-level split assignments."""
    features = data["X"]

    # These dimensions match Sam's current three-second Mel cache.
    if features.ndim != 3 or features.shape[1:] != EXPECTED_SHAPE:
        raise ValueError("X must have shape (N, 128, 130).")
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError("X must contain floating-point spectrogram values.")

    for name in REQUIRED_ARRAYS[1:]:
        values = data[name]
        if values.ndim != 1 or len(values) != len(features):
            raise ValueError(f"{name} must have one entry per segment.")
        if values.dtype.kind != "U":
            raise ValueError(f"{name} must contain strings.")
        if np.any(np.char.strip(values) == ""):
            raise ValueError(f"{name} contains an empty value.")

    if set(data["split"]) != {"train", "val", "test"}:
        raise ValueError(
            "Cache must contain only train, val, and test splits, "
            "with at least one segment in each."
        )

    # Every segment from a song must keep the same split and genre.
    song_splits = {}
    song_labels = {}
    for song, split, label in zip(
        data["song_id"], data["split"], data["label"]
    ):
        if song in song_splits and song_splits[song] != split:
            raise ValueError(f"Song {song} appears in multiple splits.")
        if song in song_labels and song_labels[song] != label:
            raise ValueError(f"Song {song} has multiple genre labels.")
        song_splits[song] = split
        song_labels[song] = label


def load_training_data(path):
    """Return train/val arrays and their shared ordered genre names."""
    with np.load(path, allow_pickle=False) as cache:
        missing = [
            name for name in REQUIRED_ARRAYS if name not in cache.files
        ]
        if missing:
            raise ValueError(
                "Missing required arrays: " + ", ".join(missing)
            )
        data = {name: cache[name] for name in REQUIRED_ARRAYS}

    check_cache(data)

    # Build the label mapping from training data only.
    train_mask = data["split"] == "train"
    genres = sorted(set(data["label"][train_mask].tolist()))
    label_numbers = {
        genre: number for number, genre in enumerate(genres)
    }
    result = {"genres": genres}

    # Leave held-out test samples out of the returned training data.
    for split in ("train", "val"):
        mask = data["split"] == split
        labels = data["label"][mask].tolist()
        if any(label not in label_numbers for label in labels):
            raise ValueError(
                f"{split} contains a genre absent from the training set."
            )

        # Keep the cached dB values and add the CNN's channel dimension.
        features = data["X"][mask].astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"{split} contains values that are not finite.")

        result[split] = {
            "X": features[..., np.newaxis],
            "y": np.array(
                [label_numbers[label] for label in labels],
                dtype=np.int64,
            ),
            "song_id": data["song_id"][mask],
            "filename": data["filename"][mask],
        }

    return result


def main():
    default_cache = (
        Path(__file__).resolve().parents[1] / "Data" / "gtzan_mels.npz"
    )
    parser = argparse.ArgumentParser(
        description="Check a Mel cache and summarize training/validation data."
    )
    parser.add_argument(
        "cache", nargs="?", type=Path, default=default_cache
    )
    args = parser.parse_args()
    data = load_training_data(args.cache)

    print("Genre order:", ", ".join(data["genres"]))
    for split in ("train", "val"):
        part = data[split]
        songs = len(set(part["song_id"]))
        print(f"{split}: {len(part['y'])} segments from {songs} songs")
        print(f"  Input shape: {part['X'].shape}")
        for number, genre in enumerate(data["genres"]):
            count = int(np.count_nonzero(part["y"] == number))
            print(f"  {genre}: {count} segments")


if __name__ == "__main__":
    main()
