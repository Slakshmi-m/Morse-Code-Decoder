import json
import os
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

from scripts.mix_datasets import abs_file, mix


def _make_dummy_dataset(root, name, n=3):
    """Write a minimal dataset directory with n WAV files and metadata.json."""
    dataset_dir = os.path.join(root, name)
    audio_dir   = os.path.join(dataset_dir, 'audio')
    os.makedirs(audio_dir)
    meta = []
    for i in range(n):
        fname = f'sample_{i:06d}.wav'
        wavfile.write(os.path.join(audio_dir, fname), 8000,
                      np.zeros(800, dtype=np.int16))
        meta.append({
            'file': fname, 'text': 'TEST',
            'wpm': 15, 'freq': 700.0, 'noise': 0.0, 'fw_char_wpm': 0,
        })
    with open(os.path.join(dataset_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)
    return dataset_dir


class TestAbsFile(unittest.TestCase):

    def test_absolute_path_unchanged(self):
        item   = {'file': '/absolute/path/audio.wav'}
        result = abs_file('/some/dir', item)
        self.assertEqual(result, '/absolute/path/audio.wav')

    def test_relative_path_resolved_under_audio_subdir(self):
        item   = {'file': 'sample_000001.wav'}
        result = abs_file('/dataset/dir', item)
        self.assertEqual(result,
                         os.path.abspath('/dataset/dir/audio/sample_000001.wav'))

    def test_result_is_always_absolute(self):
        item   = {'file': 'foo.wav'}
        result = abs_file('/some/dir', item)
        self.assertTrue(os.path.isabs(result))


class TestMix(unittest.TestCase):

    def test_combined_count_is_sum_of_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth = _make_dummy_dataset(tmp, 'synth', n=5)
            real  = _make_dummy_dataset(tmp, 'real',  n=3)
            out   = os.path.join(tmp, 'combined')
            mix(synth, real, out, real_weight=1.0)
            with open(os.path.join(out, 'metadata.json')) as f:
                combined = json.load(f)
            self.assertEqual(len(combined), 8)

    def test_real_weight_2_doubles_real_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth = _make_dummy_dataset(tmp, 'synth', n=4)
            real  = _make_dummy_dataset(tmp, 'real',  n=3)
            out   = os.path.join(tmp, 'combined')
            mix(synth, real, out, real_weight=2.0)
            with open(os.path.join(out, 'metadata.json')) as f:
                combined = json.load(f)
            self.assertEqual(len(combined), 4 + 3 * 2)

    def test_real_weight_3_triples_real_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth = _make_dummy_dataset(tmp, 'synth', n=5)
            real  = _make_dummy_dataset(tmp, 'real',  n=2)
            out   = os.path.join(tmp, 'combined')
            mix(synth, real, out, real_weight=3.0)
            with open(os.path.join(out, 'metadata.json')) as f:
                combined = json.load(f)
            self.assertEqual(len(combined), 5 + 2 * 3)

    def test_all_file_paths_are_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth = _make_dummy_dataset(tmp, 'synth', n=2)
            real  = _make_dummy_dataset(tmp, 'real',  n=2)
            out   = os.path.join(tmp, 'combined')
            mix(synth, real, out, real_weight=1.0)
            with open(os.path.join(out, 'metadata.json')) as f:
                combined = json.load(f)
            for entry in combined:
                self.assertTrue(os.path.isabs(entry['file']),
                                f"Not absolute: {entry['file']}")

    def test_metadata_fields_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth = _make_dummy_dataset(tmp, 'synth', n=2)
            real  = _make_dummy_dataset(tmp, 'real',  n=2)
            out   = os.path.join(tmp, 'combined')
            mix(synth, real, out, real_weight=1.0)
            with open(os.path.join(out, 'metadata.json')) as f:
                combined = json.load(f)
            for entry in combined:
                for field in ('file', 'text', 'wpm', 'freq', 'noise', 'fw_char_wpm'):
                    self.assertIn(field, entry, f"Missing field '{field}'")

    def test_output_metadata_is_shuffled(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth = _make_dummy_dataset(tmp, 'synth', n=20)
            real  = _make_dummy_dataset(tmp, 'real',  n=20)
            out   = os.path.join(tmp, 'combined')
            mix(synth, real, out, real_weight=1.0)
            with open(os.path.join(out, 'metadata.json')) as f:
                combined = json.load(f)
            paths = [e['file'] for e in combined]
            # With 40 entries the chance of them being in insertion order is negligible
            synth_paths = [e for e in paths if 'synth' in e]
            real_paths  = [e for e in paths if 'real'  in e]
            self.assertEqual(len(synth_paths), 20)
            self.assertEqual(len(real_paths),  20)

    def test_output_directory_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth = _make_dummy_dataset(tmp, 'synth', n=2)
            real  = _make_dummy_dataset(tmp, 'real',  n=2)
            out   = os.path.join(tmp, 'does', 'not', 'exist')
            mix(synth, real, out, real_weight=1.0)
            self.assertTrue(os.path.exists(os.path.join(out, 'metadata.json')))


if __name__ == '__main__':
    unittest.main()
