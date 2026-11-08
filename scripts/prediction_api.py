# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Accepts audio uploads and returns the top predicted genres.

from pathlib import Path
from tempfile import TemporaryDirectory
import librosa
import soundfile as sf
from datetime import datetime

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

# error log path and helper
ERROR_LOG_PATH = Path("Audio Upload Error Log") / "upload_errors.txt"
ERROR_LOG_PATH.parent.mkdir(exist_ok=True)

def log_upload_error(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

router = APIRouter()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class InvalidAudioError(ValueError):
    """The uploaded audio could not be processed."""


def predict_file(path):
    # Connect the saved model and shared preprocessing here.
    # This will return genre names and one probability row per segment.
    raise HTTPException(503, "The prediction model is not ready.")


def rank_predictions(genres, scores, n):
    genres = list(genres)
    scores = np.asarray(scores, dtype=float)

    if not genres or any(
        not isinstance(genre, str) or not genre.strip() for genre in genres
    ):
        raise ValueError("Genre names must be nonempty strings.")
    if len(set(genres)) != len(genres):
        raise ValueError("Genre names must be unique.")
    if (
        scores.ndim != 2
        or scores.shape[0] == 0
        or scores.shape[1] != len(genres)
    ):
        raise ValueError("Scores must match the segments and genres.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Scores must be between zero and one.")
    if not np.allclose(scores.sum(axis=1), 1, atol=1e-4, rtol=0):
        raise ValueError("Each segment's scores must sum to one.")

    # Average the segments before choosing the top genres.
    results = []
    for genre, score in zip(genres, scores.mean(axis=0)):
        results.append({"genre": genre, "confidence": float(score)})

    results.sort(key=lambda result: result["confidence"], reverse=True)

    # Keep the original scores; the selected results may sum to less than one.
    return {"predictions": results[:n], "segments": len(scores)}


@router.post("/predict")
def predict(file: UploadFile = File(...), n: int = Query(5, ge=1, le=8)):

    # check for file presence
    if not file or not file.filename:
        log_upload_error("No audio file was uploaded.")
        raise HTTPException(400, "No audio file was uploaded.")

    # check for file type
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp3", ".wav"}:
        log_upload_error(f"Invalid file type: {file.filename}")
        raise HTTPException(400, "Please upload an MP3 or WAV file.")

    # check for file size
    audio = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(audio) > MAX_UPLOAD_BYTES:
        log_upload_error(f"File too large: {file.filename}")
        raise HTTPException(413, "Audio file is too large (10 MB maximum).")
    if not audio:
        log_upload_error(f"Empty file: {file.filename}")
        raise HTTPException(400, "The audio file is empty.")

    # Remove the upload after prediction, including when an error occurs.
    with TemporaryDirectory() as directory:
        path = Path(directory) / ("upload" + suffix)
        path.write_bytes(audio)
        
        # upload validation

        # check for readability
        try:
            audio_sig, sample_rate = librosa.load(path, sr=None, mono=True)
        except Exception:
            log_upload_error(f"Error loading audio file: {file.filename}")
            raise HTTPException(400, "The audio file is unreadable or may be corrupted.")

        # check for appropriate length over 10 sec
        audio_length = librosa.get_duration(y=audio_sig, sr=sample_rate)
        if audio_length < 10.0:
            log_upload_error(f"Audio file too short: {file.filename}")
            raise HTTPException(400, "The audio file must be at least 10 seconds long.")

        # trim the audio to middle 30 seconds if longer than 30
        if audio_length > 30.0:
            start = (audio_length - 30.0) / 2
            end = start + 30.0
            audio_sig = audio_sig[int(start * sample_rate):int(end * sample_rate)]
            sf.write(path, audio_sig, sample_rate)

        # pass the audio to the model
        try: 
            genres, scores = predict_file(path)
        except InvalidAudioError:
            log_upload_error(f"Error processing audio file: {file.filename}")
            raise HTTPException(400, "The audio file could not be processed.") 

    try:
        result = rank_predictions(genres, scores, n)
        if n > len(genres):
            raise HTTPException(
                422, "The model has fewer genres than requested."
            )
        return result
    except (TypeError, ValueError) as error:
        raise HTTPException(
            500, "The model returned invalid predictions."
        ) from error
