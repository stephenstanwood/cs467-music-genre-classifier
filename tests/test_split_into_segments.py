# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Tests splitting audio into three-second pieces.

import unittest

import numpy as np

from scripts.preprocessing.preprocess import (
    SEGMENT_SAMPLES,
    split_into_segments,
)


class TestSplitIntoSegments(unittest.TestCase):
    def test_empty_audio_is_rejected(self):
        with self.assertRaises(ValueError):
            split_into_segments(np.array([]))

    # One sample below the default 1.5-second minimum.
    def test_audio_below_minimum_is_rejected(self):
        audio = np.ones(SEGMENT_SAMPLES // 2 - 1)
        with self.assertRaises(ValueError):
            split_into_segments(audio)

    def test_full_segment_is_unchanged(self):
        audio = np.arange(SEGMENT_SAMPLES, dtype=float)
        np.testing.assert_array_equal(split_into_segments(audio), [audio])

    def test_two_segments_stay_in_order(self):
        first = np.ones(SEGMENT_SAMPLES)
        second = np.zeros(SEGMENT_SAMPLES)
        audio = np.concatenate([first, second])

        np.testing.assert_array_equal(
            split_into_segments(audio), [first, second]
        )

    # A full segment is kept, but a tail just below half is dropped.
    def test_short_tail_is_dropped(self):
        full = np.ones(SEGMENT_SAMPLES)
        tail = np.ones(SEGMENT_SAMPLES // 2 - 1)
        audio = np.concatenate([full, tail])

        np.testing.assert_array_equal(split_into_segments(audio), [full])

    # Exactly half a segment is kept and padded with zeros.
    def test_half_segment_is_padded(self):
        half = SEGMENT_SAMPLES // 2
        audio = np.ones(half)
        expected = np.concatenate([audio, np.zeros(half)])

        np.testing.assert_array_equal(
            split_into_segments(audio), [expected]
        )


if __name__ == "__main__":
    unittest.main()