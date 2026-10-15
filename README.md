# cs467-music-genre-classifier

CS 467 capstone: a neural network that predicts the top music genres for an uploaded audio clip.

## FMA-small setup

Download `fma_small.zip` and `fma_metadata.zip` from the
[official FMA dataset page](https://github.com/mdeff/fma#data).
Extract both into `data/raw/`. The resulting paths should include:

- `data/raw/fma_metadata/tracks.csv`
- `data/raw/fma_small/000/000002.mp3`

For a new checkout, run these commands from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For later sessions, activate the existing environment with
`source .venv/bin/activate`.

## Build the manifests and audit the audio

Run these scripts in order:

```bash
python scripts/build_fma_manifest.py
python scripts/audit_fma_audio.py
python scripts/build_fma_usable_manifest.py
```

The scripts save four files in `data/manifests/`:

- `fma_small.csv`: the original 8,000 tracks with paths, genres, and official split assignments.
- `fma_small_audio_audit.csv`: decoding results, audio properties, and errors for each track.
- `fma_small_usable.csv`: the 7,994 tracks remaining after reviewed exclusions.
- `fma_small_exclusions.csv`: the six excluded tracks, with reasons and a source link.

Use `fma_small_usable.csv` for FMA preprocessing and training. Audio paths
in the manifests are relative to the project folder.

## Reviewed exclusions

Our SoundFile audit flagged tracks 98565, 98567, 98569, 99134, 108925,
and 133297. These tracks also appear in
[FMA's published issue](https://github.com/mdeff/fma/issues/8).

All six were in the training split. The usable manifest preserves the
remaining official assignments:

| Split | Tracks |
| --- | ---: |
| Training | 6,394 |
| Validation | 800 |
| Test | 800 |
| Total | 7,994 |

The original manifest and raw files are retained. The usable-manifest
script stops for review if a future audit flags a different set of tracks.

Raw datasets, processed features, and the virtual environment stay local
and are excluded from Git.
