"""
Batch-process GTZAN's raw audio into one CSV of 57 features per 3-second piece.
 
Usage:
    python build_features.py --audio-dir Data/genres_original --out features/gtzan_57_features.csv
 
Optional:
    --mels-dir features/mels   also save a mel-spectrogram .npy per piece (CNN experiment)
    --workers 4                number of CPU processes (default: all cores)
    --limit 20                 only process the first N songs (quick test run)
 
Expected folder layout (the Kaggle GTZAN download):
    genres_original/blues/blues.00000.wav
    genres_original/blues/blues.00001.wav
    ...
"""
 
import argparse
import os
from multiprocessing import Pool
from pathlib import Path
 
import numpy as np
import pandas as pd
 
import preprocess as pp
 
AUDIO_EXTS = {".wav", ".mp3", ".au", ".flac", ".ogg"}
 
 
def find_songs(audio_dir):
    """Return (path, genre) for every audio file, genre = parent folder name."""
    songs = []
    for path in sorted(Path(audio_dir).rglob("*")):
        if path.suffix.lower() in AUDIO_EXTS:
            songs.append((path, path.parent.name))
    return songs
 
 
def process_song(args):
    path, genre, mels_dir = args
    song_id = path.stem  # e.g. "blues.00000" -- keeps all pieces of a song together
    try:
        y = pp.load_audio(path)
        rows = []
        for i, seg in enumerate(pp.split_into_segments(y)):
            row = {"filename": f"{song_id}.{i}.wav", "song_id": song_id,
                   "segment": i, "label": genre}
            row.update(pp.extract_features(seg))
            rows.append(row)
            if mels_dir:
                np.save(Path(mels_dir) / f"{song_id}.{i}.npy", pp.mel_spectrogram(seg))
        return rows, None
    except Exception as e:  # corrupt / unreadable file
        return [], f"{path}: {e}"
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default="features/gtzan_57_features.csv")
    ap.add_argument("--mels-dir", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
 
    songs = find_songs(args.audio_dir)[: args.limit]
    if not songs:
        raise SystemExit(f"No audio files found under {args.audio_dir}")
    if args.mels_dir:
        Path(args.mels_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
 
    print(f"Processing {len(songs)} songs with {args.workers} workers...")
    jobs = [(p, g, args.mels_dir) for p, g in songs]
    all_rows, errors = [], []
    with Pool(args.workers) as pool:
        for n, (rows, err) in enumerate(pool.imap_unordered(process_song, jobs), 1):
            all_rows.extend(rows)
            if err:
                errors.append(err)
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} songs done")
 
    cols = ["filename", "song_id", "segment"] + pp.FEATURE_COLUMNS + ["label"]
    df = pd.DataFrame(all_rows)[cols].sort_values(["song_id", "segment"])
    df.to_csv(args.out, index=False)
 
    print(f"\nSaved {len(df)} rows from {df['song_id'].nunique()} songs to {args.out}")
    print(df["label"].value_counts().sort_index().to_string())
    if errors:
        log = Path(args.out).with_suffix(".errors.txt")
        log.write_text("\n".join(errors))
        print(f"\nSkipped {len(errors)} unreadable file(s); details in {log}")
 
 
if __name__ == "__main__":
    main()
 