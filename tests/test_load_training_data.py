# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Checks the training-data loader using a small generated cache.

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.load_training_data import load_training_data


class TestLoadTrainingData(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "mels.npz"

        # Two training songs, one validation song, and one test song.
        # The first training song has two segments.
        self.arrays = {
            "X": np.full((5, 128, 130), -40, dtype=np.float16),
            "label": np.array(
                ["rock", "jazz", "rock", "jazz", "rock"], dtype="U16"
            ),
            "song_id": np.array([
                "rock.00000", "jazz.00000", "rock.00000",
                "jazz.00001", "rock.00001",
            ]),
            "split": np.array([
                "train", "train", "train", "val", "test",
            ]),
            "filename": np.array([
                "rock.00000.0.wav",
                "jazz.00000.0.wav",
                "rock.00000.1.wav",
                "jazz.00001.0.wav",
                "rock.00001.0.wav",
            ]),
        }

    def load_data(self):
        np.savez_compressed(self.path, **self.arrays)
        return load_training_data(self.path)

    def test_training_shape_includes_channel_dimension(self):
        data = self.load_data()
        self.assertEqual(data["train"]["X"].shape, (3, 128, 130, 1))

    def test_training_labels_follow_genre_order(self):
        data = self.load_data()
        self.assertEqual(data["train"]["y"].tolist(), [1, 0, 1])

    def test_genre_order_uses_training_labels_only(self):
        self.arrays["label"][4] = "blues"
        data = self.load_data()
        self.assertEqual(data["genres"], ["jazz", "rock"])

    def test_validation_keeps_the_original_segment_values(self):
        self.arrays["X"][3].fill(-17.5)
        data = self.load_data()
        self.assertEqual(data["val"]["X"][0, 0, 0, 0], -17.5)

    def test_test_split_is_not_returned(self):
        data = self.load_data()
        self.assertNotIn("test", data)

    def test_song_cannot_cross_training_and_validation(self):
        self.arrays["song_id"][3] = "rock.00000"
        self.arrays["label"][3] = "rock"
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            self.load_data()

    def test_song_cannot_cross_training_and_test(self):
        self.arrays["song_id"][4] = "rock.00000"
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            self.load_data()

    def test_song_cannot_have_conflicting_genre_labels(self):
        self.arrays["label"][2] = "jazz"
        with self.assertRaisesRegex(ValueError, "multiple genre labels"):
            self.load_data()

    def test_validation_genre_must_exist_in_training(self):
        self.arrays["label"][3] = "blues"
        with self.assertRaisesRegex(ValueError, "training set"):
            self.load_data()

    def test_wrong_spectrogram_shape_is_rejected(self):
        self.arrays["X"] = np.zeros((5, 128, 129), dtype=np.float16)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.load_data()

    def test_metadata_must_match_the_segment_count(self):
        self.arrays["filename"] = self.arrays["filename"][:4]
        with self.assertRaisesRegex(ValueError, "one entry"):
            self.load_data()

    def test_missing_required_array_is_rejected(self):
        del self.arrays["filename"]
        with self.assertRaisesRegex(ValueError, "Missing required arrays"):
            self.load_data()

    def test_unknown_split_is_rejected(self):
        self.arrays["split"][4] = "other"
        with self.assertRaisesRegex(ValueError, "splits"):
            self.load_data()

    def test_nonfinite_training_values_are_rejected(self):
        self.arrays["X"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.load_data()


if __name__ == "__main__":
    unittest.main()
