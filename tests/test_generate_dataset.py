import json
import os
import tempfile
import unittest

import numpy as np

from scripts.generate_dataset import text_to_audio, MORSE_TABLE, SAMPLE_RATE


class TestTextToAudio(unittest.TestCase):

    def test_output_is_int16(self):
        audio = text_to_audio('E', wpm=15, freq=700, noise=0.0)
        self.assertEqual(audio.dtype, np.int16)

    def test_output_is_non_empty(self):
        audio = text_to_audio('SOS', wpm=15, freq=700, noise=0.0)
        self.assertGreater(len(audio), 0)

    def test_longer_text_is_longer_audio(self):
        short = text_to_audio('E',     wpm=15, freq=700, noise=0.0)
        long_ = text_to_audio('HELLO', wpm=15, freq=700, noise=0.0)
        self.assertGreater(len(long_), len(short))

    def test_faster_wpm_is_shorter_audio(self):
        slow = text_to_audio('HELLO', wpm=5,  freq=700, noise=0.0)
        fast = text_to_audio('HELLO', wpm=25, freq=700, noise=0.0)
        self.assertGreater(len(slow), len(fast))

    def test_noise_changes_audio(self):
        clean = text_to_audio('SOS', wpm=15, freq=700, noise=0.0)
        noisy = text_to_audio('SOS', wpm=15, freq=700, noise=0.5)
        self.assertFalse(np.array_equal(clean, noisy))

    def test_unknown_chars_produce_silence_fallback(self):
        audio = text_to_audio('@@@', wpm=15, freq=700, noise=0.0)
        self.assertEqual(len(audio), SAMPLE_RATE)  # 1-second silence fallback

    def test_farnsworth_gives_longer_audio_than_normal(self):
        normal = text_to_audio('AB', wpm=15, freq=700, noise=0.0)
        fw     = text_to_audio('AB', wpm=5,  freq=700, noise=0.0, fw_char_wpm=15)
        self.assertGreater(len(fw), len(normal))

    def test_all_morse_table_chars_produce_audio(self):
        for char in MORSE_TABLE:
            audio = text_to_audio(char, wpm=15, freq=700, noise=0.0)
            self.assertGreater(len(audio), 0,
                               f"Empty audio for '{char}'")

    def test_amplitude_within_int16_range(self):
        audio = text_to_audio('SOS', wpm=15, freq=700, noise=0.3)
        self.assertLessEqual(np.max(np.abs(audio.astype(np.int32))), 32767)

    def test_word_with_space(self):
        audio = text_to_audio('CQ DE', wpm=15, freq=700, noise=0.0)
        self.assertGreater(len(audio), 0)

    def test_different_frequencies_produce_different_audio(self):
        a = text_to_audio('E', wpm=15, freq=600, noise=0.0)
        b = text_to_audio('E', wpm=15, freq=900, noise=0.0)
        self.assertFalse(np.array_equal(a, b))


class TestGenerateDatasetIntegration(unittest.TestCase):

    def test_creates_metadata_and_audio_files(self):
        from scripts.generate_dataset import generate_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(output_dir=tmpdir, n_random=5)
            meta_path = os.path.join(tmpdir, 'metadata.json')
            self.assertTrue(os.path.exists(meta_path))
            audio_dir = os.path.join(tmpdir, 'audio')
            self.assertTrue(os.path.isdir(audio_dir))

    def test_metadata_has_required_fields(self):
        from scripts.generate_dataset import generate_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(output_dir=tmpdir, n_random=2)
            with open(os.path.join(tmpdir, 'metadata.json')) as f:
                meta = json.load(f)
            for entry in meta[:10]:
                for field in ('file', 'text', 'wpm', 'freq', 'noise', 'fw_char_wpm'):
                    self.assertIn(field, entry, f"Missing field '{field}'")

    def test_total_count_includes_guaranteed_and_random(self):
        from scripts.generate_dataset import generate_dataset, ALL_CHARS
        n_random = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_dataset(output_dir=tmpdir, n_random=n_random)
            with open(os.path.join(tmpdir, 'metadata.json')) as f:
                meta = json.load(f)
            guaranteed = len(ALL_CHARS) * 6 * 2  # 6 WPMs × 2 freqs
            self.assertEqual(len(meta), guaranteed + n_random)


if __name__ == '__main__':
    unittest.main()
