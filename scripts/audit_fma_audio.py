# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Checks FMA-small audio files and saves an audit report.

from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def check_audio(path):
    """Read one audio file and record its properties and any errors."""
    result = {
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
        "reported_frames": None,
        "decoded_frames": 0,
        "error": "",
    }

    try:
        with sf.SoundFile(path) as audio:
            result["sample_rate"] = audio.samplerate
            result["channels"] = audio.channels
            result["reported_frames"] = audio.frames

            # Read in small blocks so we do not load the whole file into memory.
            while True:
                block = audio.read(65536, dtype="float32")
                if len(block) == 0:
                    break
                result["decoded_frames"] += len(block)

            result["duration_seconds"] = (
                result["decoded_frames"] / result["sample_rate"]
            )

            # Flag differences for review before deciding to exclude anything.
            if result["decoded_frames"] == 0:
                result["error"] = "No audio samples decoded."
            elif result["decoded_frames"] != result["reported_frames"]:
                result["error"] = (
                    f"Decoded {result['decoded_frames']} frames; "
                    f"file reports {result['reported_frames']}."
                )

    except (sf.SoundFileError, OSError, ValueError) as error:
        # Keep checking the remaining files if this one cannot be read.
        result["error"] = str(error)

    return result


def main():
    """Check each track in the manifest and save a separate audit report."""
    manifest = pd.read_csv(ROOT / "data/manifests/fma_small.csv")
    rows = []

    for count, row in enumerate(manifest.to_dict("records"), start=1):
        row.update(check_audio(ROOT / row["path"]))
        rows.append(row)

        if count % 500 == 0 or count == len(manifest):
            print(f"Checked {count}/{len(manifest)} files", flush=True)

    # Preserve the original manifest and its official training/validation/test split.
    report = pd.DataFrame(rows)
    output = ROOT / "data/manifests/fma_small_audio_audit.csv"
    report.to_csv(output, index=False)

    flagged = report[report["error"] != ""]
    print(f"\nFiles checked: {len(report)}")
    print(f"Flagged files: {len(flagged)}")
    print(f"Saved report: {output}")
    print("\nDecoded duration in seconds:")
    print(report["duration_seconds"].describe().to_string())

    if not flagged.empty:
        print("\nFirst 10 flagged files:")
        print(flagged[["track_id", "error"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
