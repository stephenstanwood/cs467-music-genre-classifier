# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Tests prediction requests without a trained model.

from io import BytesIO
import unittest
import wave
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import app
from scripts.prediction_api import InvalidAudioError


class TestPredictionAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        # Use ten seconds of valid audio to pass the upload checks.
        audio = BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\x00\x00" * 80000)
        self.upload = {"file": ("clip.wav", audio.getvalue(), "audio/wav")}

        # Keep test errors out of the upload log.
        log_patch = patch("scripts.prediction_api.log_upload_error")
        log_patch.start()
        self.addCleanup(log_patch.stop)
        self.genres = ["blues", "jazz", "rock", "pop", "metal"]

        # Blues leads in the first segment; jazz leads after averaging both.
        self.scores = np.array([
            [0.5, 0.125, 0.125, 0.125, 0.125],
            [0.0, 0.625, 0.125, 0.125, 0.125],
        ])

        self.model_patch = patch("scripts.prediction_api.predict_file")
        self.model = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.model.return_value = (self.genres, self.scores)

    def test_results_use_average_scores_without_rescaling(self):
        response = self.client.post("/predict?n=2", files=self.upload)
        self.assertEqual(response.json(), {
            "predictions": [
                {"genre": "jazz", "confidence": 0.375},
                {"genre": "blues", "confidence": 0.25},
            ],
            "segments": 2,
        })

    def test_empty_genres_are_a_server_error(self):
        self.model.return_value = ([], np.empty((1, 0)))
        response = self.client.post("/predict", files=self.upload)
        self.assertEqual(response.status_code, 500)

    def test_default_returns_five_genres(self):
        response = self.client.post("/predict", files=self.upload)
        self.assertEqual(len(response.json()["predictions"]), 5)

    def test_zero_results_are_rejected(self):
        response = self.client.post("/predict?n=0", files=self.upload)
        self.assertEqual(response.status_code, 422)

    def test_more_than_eight_results_are_rejected(self):
        response = self.client.post("/predict?n=9", files=self.upload)
        self.assertEqual(response.status_code, 422)

    def test_missing_model_returns_503(self):
        self.model_patch.stop()
        response = self.client.post("/predict", files=self.upload)
        self.assertEqual(response.status_code, 503)

    def test_missing_upload_is_rejected(self):
        response = self.client.post("/predict")
        self.assertEqual(response.status_code, 422)

    def test_empty_upload_is_rejected(self):
        response = self.client.post(
            "/predict", files={"file": ("clip.wav", b"", "audio/wav")}
        )
        self.assertEqual(response.status_code, 400)

    def test_unsupported_file_type_is_rejected(self):
        response = self.client.post(
            "/predict", files={"file": ("notes.txt", b"text", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)

    def test_oversized_upload_is_rejected(self):
        # Use a small limit to avoid creating a large test file.
        with patch("scripts.prediction_api.MAX_UPLOAD_BYTES", 4):
            response = self.client.post("/predict", files=self.upload)
        self.assertEqual(response.status_code, 413)

    def test_nan_scores_are_rejected(self):
        self.scores[0, 0] = np.nan
        response = self.client.post("/predict", files=self.upload)
        self.assertEqual(response.status_code, 500)

    def test_scores_must_sum_to_one(self):
        self.scores[0] *= 0.5
        response = self.client.post("/predict", files=self.upload)
        self.assertEqual(response.status_code, 500)

    def test_unreadable_audio_returns_400(self):
        self.model.side_effect = InvalidAudioError()
        response = self.client.post("/predict", files=self.upload)
        self.assertEqual(response.status_code, 400)

    def test_upload_is_removed_after_prediction(self):
        self.client.post("/predict", files=self.upload)
        path = self.model.call_args.args[0]
        self.assertFalse(path.exists())

    def test_upload_is_removed_after_an_error(self):
        self.model.side_effect = InvalidAudioError()
        self.client.post("/predict", files=self.upload)
        path = self.model.call_args.args[0]
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
