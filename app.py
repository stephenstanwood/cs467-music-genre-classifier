"""Small Python server for the Top N Genre Music Classifier interface."""

from io import BytesIO
from pathlib import Path
import tempfile

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # Draw images without opening a desktop window.
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Top N Genre Music Classifier")
BASE_DIR = Path(__file__).resolve().parent
# This folder contains index.html, styles.css, and script.js.
FRONTEND_DIR = BASE_DIR / "frontend"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB


@app.post("/spectrogram")
async def create_spectrogram(file: UploadFile = File(...)):
    """Read an audio upload and return its Mel spectrogram as a PNG."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Please upload an MP3, WAV, FLAC, or OGG file.")

    # Read only up to the size limit so an oversized file is rejected.
    audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is too large (30 MB maximum).")
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The audio file is empty.")

    # Save the upload briefly because Librosa can load a file by its path.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_path = Path(temporary_file.name)

        # Use only the first 20 seconds, converted to mono at 22,050 Hz.
        audio, sample_rate = librosa.load(
            str(temporary_path), sr=22050, mono=True, duration=20.0
        )
        if len(audio) < 2048:
            raise ValueError("Audio clip is too short. Please choose a longer file.")

        # Turn the audio into 128 Mel-frequency bands over time.
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128
        )
        # Convert sound energy to decibels so the colors are easier to see.
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)

        # Draw the spectrogram and save it in memory as a PNG image.
        fig, ax = plt.subplots(figsize=(10, 4))
        plot = librosa.display.specshow(
            mel_db, sr=sample_rate, hop_length=512,
            x_axis="time", y_axis="mel", cmap="magma", ax=ax
        )
        fig.colorbar(plot, ax=ax, format="%+d dB")
        ax.set_title("Mel Spectrogram")
        fig.tight_layout()
        image = BytesIO()
        fig.savefig(image, format="png", dpi=120)
        plt.close(fig)
        return Response(content=image.getvalue(), media_type="image/png")
    except (ValueError, RuntimeError, EOFError) as error:
        raise HTTPException(status_code=400, detail=f"Unable to read audio: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


# Serve the HTML, CSS, and JavaScript from this same Python server.
# Keep this last so it does not interfere with /spectrogram.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")