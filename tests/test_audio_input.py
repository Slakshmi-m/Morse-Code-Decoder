import os
import unittest
import numpy as np
from scipy.io import wavfile

from src.audio_input import AudioInput, _SD_AVAILABLE, _PAW_AVAILABLE

SR = 8000
TEST_AUDIO = os.path.join(os.path.dirname(__file__), '..', 'data', 'test')


class TestAudioInputInit(unittest.TestCase):

    def test_default_sample_rate(self):
        ai = AudioInput()
        self.assertEqual(ai.sample_rate, SR)

    def test_default_chunk_duration(self):
        ai = AudioInput()
        self.assertAlmostEqual(ai.chunk_duration, 0.1)

    def test_chunk_size_matches_rate_and_duration(self):
        ai = AudioInput(sample_rate=8000, chunk_duration=0.1)
        self.assertEqual(ai.chunk_size, 800)

    def test_custom_params(self):
        ai = AudioInput(sample_rate=16000, chunk_duration=0.2)
        self.assertEqual(ai.sample_rate, 16000)
        self.assertEqual(ai.chunk_size, 3200)

    def test_not_running_initially(self):
        ai = AudioInput()
        self.assertFalse(ai._running)


class TestCallbacks(unittest.TestCase):

    def test_register_single_callback(self):
        ai     = AudioInput()
        called = []
        ai.register_callback(lambda s, r: called.append(len(s)))
        ai._dispatch(np.zeros(800, dtype=np.int16))
        self.assertEqual(called, [800])

    def test_register_multiple_callbacks(self):
        ai     = AudioInput()
        counts = []
        ai.register_callback(lambda s, r: counts.append('a'))
        ai.register_callback(lambda s, r: counts.append('b'))
        ai._dispatch(np.zeros(100, dtype=np.int16))
        self.assertEqual(counts, ['a', 'b'])

    def test_callback_receives_correct_sample_rate(self):
        ai    = AudioInput(sample_rate=SR)
        rates = []
        ai.register_callback(lambda s, r: rates.append(r))
        ai._dispatch(np.zeros(10, dtype=np.int16))
        self.assertEqual(rates, [SR])

    def test_callback_exception_does_not_crash_dispatch(self):
        ai = AudioInput()
        ai.register_callback(lambda s, r: (_ for _ in ()).throw(RuntimeError('boom')))
        ai._dispatch(np.zeros(10, dtype=np.int16))  # must not raise


class TestResample(unittest.TestCase):

    def test_same_rate_returns_identical(self):
        data   = np.arange(100, dtype=np.float32)
        result = AudioInput._resample(data, 8000, 8000)
        np.testing.assert_array_equal(result, data)

    def test_upsample_doubles_length(self):
        data   = np.arange(100, dtype=np.float32)
        result = AudioInput._resample(data, 8000, 16000)
        self.assertEqual(len(result), 200)

    def test_downsample_halves_length(self):
        data   = np.arange(200, dtype=np.float32)
        result = AudioInput._resample(data, 16000, 8000)
        self.assertEqual(len(result), 100)

    def test_preserves_dtype(self):
        data   = np.zeros(100, dtype=np.int16)
        result = AudioInput._resample(data, 8000, 4000)
        self.assertEqual(result.dtype, np.int16)


class TestStreamFile(unittest.TestCase):

    def test_streams_wav_file(self):
        wav_path = os.path.join(TEST_AUDIO, 'WHO.wav')
        if not os.path.exists(wav_path):
            self.skipTest('WHO.wav not found in data/test/')
        chunks = []
        ai = AudioInput()
        ai.register_callback(lambda s, r: chunks.append(s.copy()))
        ai.stream_file(wav_path, realtime=False)
        self.assertGreater(len(chunks), 0)

    def test_chunks_are_numpy_arrays(self):
        wav_path = os.path.join(TEST_AUDIO, 'WHO.wav')
        if not os.path.exists(wav_path):
            self.skipTest('WHO.wav not found in data/test/')
        chunks = []
        ai = AudioInput()
        ai.register_callback(lambda s, r: chunks.append(s))
        ai.stream_file(wav_path, realtime=False)
        for c in chunks:
            self.assertIsInstance(c, np.ndarray)

    def test_missing_file_does_not_raise(self):
        ai = AudioInput()
        ai.register_callback(lambda s, r: None)
        ai.stream_file('/nonexistent/file.wav', realtime=False)  # must not raise

    def test_stop_when_not_running_does_not_raise(self):
        ai = AudioInput()
        ai.stop()


if __name__ == '__main__':
    unittest.main()
