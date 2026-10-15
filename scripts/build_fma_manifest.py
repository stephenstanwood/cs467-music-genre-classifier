# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Builds the FMA-small manifest using the official data split.

"""Build an FMA-small manifest; audio paths are relative to the repo root."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    # FMA's metadata CSV has two header rows for its grouped columns.
    tracks = pd.read_csv(
        ROOT / "data/raw/fma_metadata/tracks.csv",
        index_col=0,
        header=[0, 1],
        low_memory=False,
    )
    small = tracks.loc[tracks[("set", "subset")] == "small"].sort_index()

    # Keep FMA's genre labels and original training/validation/test split.
    manifest = pd.DataFrame({
        "track_id": small.index,
        # MP3 names have six digits; folders use the first three.
        "path": [
            f"data/raw/fma_small/{tid // 1000:03d}/{tid:06d}.mp3"
            for tid in small.index
        ],
        "genre": small[("track", "genre_top")].to_numpy(),
        "split": small[("set", "split")].to_numpy(),
    })

    if len(manifest) != 8000 or not manifest["track_id"].is_unique:
        raise ValueError("Expected 8,000 unique track IDs.")
    if manifest[["genre", "split"]].isna().any().any():
        raise ValueError("Some tracks are missing a genre or split.")
    if not manifest["split"].isin(["training", "validation", "test"]).all():
        raise ValueError("Unexpected split assignment.")

    # This checks file paths. The audio audit will check decoding.
    missing = [p for p in manifest["path"] if not (ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} files: {missing[:5]}")

    output = ROOT / "data/manifests/fma_small.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)

    print(f"Saved {len(manifest)} tracks to {output.relative_to(ROOT)}")
    print(f"Genres: {manifest['genre'].nunique()}")
    print(manifest["split"].value_counts().to_string())


if __name__ == "__main__":
    main()