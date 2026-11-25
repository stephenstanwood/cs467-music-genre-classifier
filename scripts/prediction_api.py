# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Accepts audio uploads and returns the top predicted genres.

import logging
from datetime import datetime
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from scripts.preprocessing.preprocess import mels_for_file

ERROR_LOG_PATH = Path("Audio Upload Error Log") / "upload_errors.txt"
logger = logging.getLogger(__name__)


def log_upload_error(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ERROR_LOG_PATH.parent.mkdir(exist_ok=True)
        with ERROR_LOG_PATH.open("a") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")
    except OSError:
        logger.warning("%s", message)

router = APIRouter()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class InvalidAudioError(ValueError):
    """The uploaded audio could not be processed."""


@cache
def load_prediction_model():
    from tensorflow.keras.models import load_model

    from models.model_loader import GENRE_LABELS, MODEL_FOLDER

    model = load_model(MODEL_FOLDER / "mel_cnn_v7.keras", compile=False)
    return model, GENRE_LABELS.tolist()


def predict_file(path):
    try:
        model, genres = load_prediction_model()
    except (ImportError, OSError, ValueError) as error:
        raise HTTPException(
            503, "The prediction model is not ready."
        ) from error

    try:
        mels = mels_for_file(path)
    except ValueError as error:
        raise InvalidAudioError() from error

    # Add the channel axis the model expects.
    scores = model(mels[..., np.newaxis], training=False).numpy()
    return genres, scores


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

    if not file or not file.filename:
        log_upload_error("No audio file was uploaded.")
        raise HTTPException(400, "No audio file was uploaded.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp3", ".wav"}:
        log_upload_error(f"Invalid file type: {file.filename}")
        raise HTTPException(400, "Please upload an MP3 or WAV file.")

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

        try:
            audio_sig, sample_rate = librosa.load(path, sr=None, mono=True)
        except Exception:
            log_upload_error(f"Error loading audio file: {file.filename}")
            raise HTTPException(400, "The audio file is unreadable or may be corrupted.")

        audio_length = librosa.get_duration(y=audio_sig, sr=sample_rate)
        if audio_length < 10.0:
            log_upload_error(f"Audio file too short: {file.filename}")
            raise HTTPException(400, "The audio file must be at least 10 seconds long.")

        # The shared preprocessing takes the middle 30 seconds.
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
