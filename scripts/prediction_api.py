# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Accepts audio uploads and returns the top predicted genres.

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

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
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp3", ".wav"}:
        raise HTTPException(400, "Please upload an MP3 or WAV file.")

    audio = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(audio) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file is too large (10 MB maximum).")
    if not audio:
        raise HTTPException(400, "The audio file is empty.")

    # Remove the upload after prediction, including when an error occurs.
    with TemporaryDirectory() as directory:
        path = Path(directory) / ("upload" + suffix)
        path.write_bytes(audio)
        try:
            genres, scores = predict_file(path)
        except InvalidAudioError as error:
            raise HTTPException(
                400, "This audio file could not be processed."
            ) from error

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
