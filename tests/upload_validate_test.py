# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Tests the upload checks on POST /predict.

# Written by Dawson Herrington to test upload validation and error handling

# imports
import io
import unittest

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts import prediction_api as api

# test version of the app
app = FastAPI()
app.include_router(api.router)
client = TestClient(app)


# generate a test wav file (a beep) of a set length
def create_test_wav(seconds):
    sample_rate = 8000
    beep = 0.1 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * sample_rate)) / sample_rate)
    buffer = io.BytesIO()
    sf.write(buffer, beep, sample_rate, format="WAV")
    return buffer.getvalue()


# send a file to the predict endpoint
def upload_file(filename, data):
    return client.post("/predict", files={"file": (filename, data)})


class TestUploadValidation(unittest.TestCase):

    # model isn't connected yet, so a fake one stands in
    # also records the audio length for the trim test
    def setUp(self):
        self.real_predict_file = api.predict_file
        self.model_saw = {}

        def fake_predict_file(path):
            self.model_saw["duration"] = sf.info(path).duration
            genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop"]
            return genres, np.full((1, 8), 1 / 8)

        api.predict_file = fake_predict_file

    # restore the real model after each test
    def tearDown(self):
        api.predict_file = self.real_predict_file

    # check for no file
    def test_no_file_uploaded(self):
        response = client.post("/predict")
        self.assertEqual(response.status_code, 422)

    # check for wrong file type
    def test_invalid_file_type(self):
        response = upload_file("song.txt", b"not audio")
        self.assertEqual(response.status_code, 400)
        self.assertIn("MP3 or WAV", response.json()["detail"])

    # mp3 passes the type check - junk audio so it fails later as unreadable
    def test_mp3_accepted(self):
        response = upload_file("song.mp3", b"junk" * 100)
        self.assertIn("unreadable", response.json()["detail"])

    # check for files over 10 MB
    def test_file_too_large(self):
        response = upload_file("song.wav", b"0" * (api.MAX_UPLOAD_BYTES + 1))
        self.assertEqual(response.status_code, 413)
        self.assertIn("10 MB", response.json()["detail"])

    # check for empty files
    def test_empty_file(self):
        response = upload_file("song.wav", b"")
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"])

    # check for corrupted files
    def test_corrupted_file(self):
        response = upload_file("song.wav", b"RIFF" + b"garbage" * 200)
        self.assertEqual(response.status_code, 400)
        self.assertIn("unreadable", response.json()["detail"])

    # check for audio under 10 seconds
    def test_audio_too_short(self):
        response = upload_file("song.wav", create_test_wav(5))
        self.assertEqual(response.status_code, 400)
        self.assertIn("at least 10 seconds", response.json()["detail"])

    # exactly 10 seconds is the minimum and passes
    def test_ten_second_minimum(self):
        response = upload_file("song.wav", create_test_wav(10))
        self.assertEqual(response.status_code, 200)

    # longer audio is trimmed to the middle 30 seconds
    def test_trim_to_middle_30(self):
        upload_file("song.wav", create_test_wav(60))
        self.assertAlmostEqual(self.model_saw["duration"], 30, delta=0.01)

    # valid upload goes through and returns the top 5
    def test_valid_upload(self):
        response = upload_file("song.wav", create_test_wav(15))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["predictions"]), 5)

    # failed uploads never reach the model
    def test_failed_upload_not_sent_to_model(self):
        upload_file("song.wav", create_test_wav(5))
        self.assertNotIn("duration", self.model_saw)


if __name__ == "__main__":
    unittest.main(verbosity=2)