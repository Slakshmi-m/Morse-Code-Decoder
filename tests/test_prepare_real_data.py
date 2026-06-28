import json
import os
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

from scripts.prepare_real_data import label_from_filename, prepare, _load_wav


class TestLabelFromFilename(unittest.TestCase):

    def test_simple_word(self):
        self.assertEqual(label_from_filename('the.mp3'), 'THE')

    def test_uppercase(self):
        self.assertEqual(label_from_filename('morse.wav'), 'MORSE')

    def test_underscore_becomes_space(self):
        self.assertEqual(label_from_filename('cq_de.mp3'), 'CQ DE')

    def test_trailing_wpm_tag_stripped_underscore(self):
        self.assertEqual(label_from_filename('hello_20wpm.mp3'), 'HELLO')

    def test_trailing_wpm_tag_stripped_dash(self):
        self.assertEqual(label_from_filename('sos-15wpm.wav'), 'SOS')

    def test_three_digit_wpm_tag_stripped(self):
        self.assertEqual(label_from_filename('cq_de-100wpm.wav'), 'CQ DE')

    def test_digits_preserved(self):
        self.assertEqual(label_from_filename('73.mp3'), '73')

    def test_path_prefix_ignored(self):
        self.assertEqual(label_from_filename('/some/dir/hello.wav'), 'HELLO')

    def test_invalid_chars_stripped(self):
        result = label_from_filename('hello123.wav')
        self.assertTrue(all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ' for c in result))

    def test_empty_after_stripping_returns_empty(self):
        result = label_from_filename('___20wpm.wav')
        self.assertEqual(result, '')


class TestLoadWav(unittest.TestCase):

    def _write_wav(self, path, data, sr=8000):
        wavfile.write(path, sr, data)

    def test_loads_int16_wav(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'test.wav')
            self._write_wav(path, np.zeros(8000, dtype=np.int16))
            data, sr = _load_wav(path)
            self.assertEqual(data.dtype, np.int16)
            self.assertEqual(len(data), 8000)

    def test_converts_float32_wav_to_int16(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'float.wav')
            self._write_wav(path, np.zeros(8000, dtype=np.float32))
            data, _ = _load_wav(path)
            self.assertEqual(data.dtype, np.int16)

    def test_stereo_returns_first_channel(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'stereo.wav')
            stereo = np.zeros((8000, 2), dtype=np.int16)
            stereo[:, 0] = 100
            self._write_wav(path, stereo)
            data, _ = _load_wav(path)
            self.assertEqual(data.ndim, 1)


class TestPrepare(unittest.TestCase):

    def test_processes_wav_files(self):
        with tempfile.TemporaryDirectory() as src, \
             tempfile.TemporaryDirectory() as out:
            wavfile.write(os.path.join(src, 'hello.wav'), 8000,
                          np.zeros(8000, dtype=np.int16))
            prepare(src, out)
            meta_path = os.path.join(out, 'metadata.json')
            self.assertTrue(os.path.exists(meta_path))
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(len(meta), 1)
            self.assertEqual(meta[0]['text'], 'HELLO')

    def test_metadata_source_is_real(self):
        with tempfile.TemporaryDirectory() as src, \
             tempfile.TemporaryDirectory() as out:
            wavfile.write(os.path.join(src, 'sos.wav'), 8000,
                          np.zeros(8000, dtype=np.int16))
            prepare(src, out)
            with open(os.path.join(out, 'metadata.json')) as f:
                meta = json.load(f)
            self.assertEqual(meta[0]['source'], 'real')
            self.assertEqual(meta[0]['noise'],  -1.0)

    def test_output_wav_at_target_sample_rate(self):
        with tempfile.TemporaryDirectory() as src, \
             tempfile.TemporaryDirectory() as out:
            wavfile.write(os.path.join(src, 'test.wav'), 16000,
                          np.zeros(16000, dtype=np.int16))
            prepare(src, out)
            with open(os.path.join(out, 'metadata.json')) as f:
                meta = json.load(f)
            out_path = os.path.join(out, 'audio', meta[0]['file'])
            sr, _ = wavfile.read(out_path)
            self.assertEqual(sr, 8000)

    def test_skips_files_with_empty_label(self):
        with tempfile.TemporaryDirectory() as src, \
             tempfile.TemporaryDirectory() as out:
            # File whose name yields no valid label
            wavfile.write(os.path.join(src, '___20wpm.wav'), 8000,
                          np.zeros(8000, dtype=np.int16))
            prepare(src, out)
            meta_path = os.path.join(out, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self.assertEqual(len(meta), 0)

    def test_empty_directory_does_not_crash(self):
        with tempfile.TemporaryDirectory() as src, \
             tempfile.TemporaryDirectory() as out:
            prepare(src, out)  # must not raise

    def test_multiple_files_all_processed(self):
        with tempfile.TemporaryDirectory() as src, \
             tempfile.TemporaryDirectory() as out:
            for word in ('alpha', 'bravo', 'charlie'):
                wavfile.write(os.path.join(src, f'{word}.wav'), 8000,
                              np.zeros(8000, dtype=np.int16))
            prepare(src, out)
            with open(os.path.join(out, 'metadata.json')) as f:
                meta = json.load(f)
            self.assertEqual(len(meta), 3)


if __name__ == '__main__':
    unittest.main()
