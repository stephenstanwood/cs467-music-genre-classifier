# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Builds a usable FMA manifest and documents reviewed exclusions.

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# These six clips failed our audit and appear in FMA's published issue.
EXCLUDED_TRACKS = {98565, 98567, 98569, 99134, 108925, 133297}
SOURCE = "https://github.com/mdeff/fma/issues/8"


def main():
    folder = ROOT / "data/manifests"
    manifest = pd.read_csv(folder / "fma_small.csv")

    # Keep empty error fields as empty strings rather than missing values.
    audit = pd.read_csv(
        folder / "fma_small_audio_audit.csv",
        keep_default_na=False,
    )

    columns = ["track_id", "path", "genre", "split"]
    if not audit[columns].equals(manifest[columns]):
        raise ValueError("The audit does not match the current manifest.")

    # Stop for review if a later audit finds a different set of problems.
    flagged_ids = set(audit.loc[audit["error"] != "", "track_id"])
    if flagged_ids != EXCLUDED_TRACKS:
        raise ValueError("Flagged tracks changed. Review them before continuing.")

    excluded_rows = manifest["track_id"].isin(EXCLUDED_TRACKS)
    usable = manifest.loc[~excluded_rows].copy()
    excluded = manifest.loc[excluded_rows].copy()

    excluded["reason"] = (
        "Known short or unreadable FMA clip; failed the SoundFile audit."
    )
    excluded["source"] = SOURCE

    # Save separate files while preserving the original split assignments.
    usable.to_csv(folder / "fma_small_usable.csv", index=False)
    excluded.to_csv(folder / "fma_small_exclusions.csv", index=False)

    print(f"Usable tracks: {len(usable)}")
    print(f"Excluded tracks: {len(excluded)}")
    print(usable["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
