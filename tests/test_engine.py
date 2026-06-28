import os
import unittest
import numpy as np
from scipy.io import wavfile

from src.engine import MorseEngine, decode_wav

SR = 8000
TEST_AUDIO = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_audio')


def _sine_wave(freq=700, duration=1.0, sr=SR, amplitude=16384):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.int16)


def _morse_audio(text, wpm=15, freq=700):
    from scripts.generate_dataset import text_to_audio
    return text_to_audio(text, wpm=wpm, freq=freq, noise=0.0)


class TestMorseEngineInit(unittest.TestCase):

    def test_detects_frequency_of_sine_wave(self):
        audio = _sine_wave(freq=700, duration=2.0)
        engine = MorseEngine(SR, audio)
        self.assertAlmostEqual(engine.detected_freq, 700, delta=50)

    def test_handles_stereo_by_taking_first_channel(self):
        mono  = _sine_wave(700, 1.0)
        stereo = np.column_stack([mono, mono])
        engine = MorseEngine(SR, stereo)
        self.assertIsNotNone(engine.detected_freq)

    def test_snr_clean_signal_is_high(self):
        audio = _sine_wave(700, 2.0)
        engine = MorseEngine(SR, audio)
        self.assertGreater(engine.snr_db, 10.0)

    def test_snr_silence_is_zero_or_low(self):
        silence = np.zeros(SR * 2, dtype=np.int16)
        engine  = MorseEngine(SR, silence)
        self.assertLess(engine.snr_db, 5.0)


class TestMorseEngineDecode(unittest.TestCase):

    def test_silence_returns_status_message(self):
        silence = np.zeros(SR * 2, dtype=np.int16)
        engine  = MorseEngine(SR, silence)
        result  = engine.decode()
        self.assertTrue(result.startswith('['))

    def test_pure_noise_returns_status_message(self):
        rng   = np.random.default_rng(42)
        noise = rng.integers(-500, 500, SR * 2, dtype=np.int16)
        engine = MorseEngine(SR, noise)
        result = engine.decode()
        self.assertIsInstance(result, str)

    def test_decode_cq(self):
        # 'CQ' produces clearly distinct dits and dahs with enough samples for SNR
        audio  = _morse_audio('CQ', wpm=15, freq=700)
        engine = MorseEngine(SR, audio)
        result = engine.decode()
        self.assertIn('CQ', result)

    def test_decode_sos(self):
        wav_path = os.path.join(TEST_AUDIO, 'test_sos.wav')
        if not os.path.exists(wav_path):
            self.skipTest('test_sos.wav not found in data/test_audio/')
        result = decode_wav(wav_path)
        self.assertIn('SOS', result)

    def test_decode_hello(self):
        wav_path = os.path.join(TEST_AUDIO, 'test_hello.wav')
        if not os.path.exists(wav_path):
            self.skipTest('test_hello.wav not found in data/test_audio/')
        result = decode_wav(wav_path)
        self.assertIn('HELLO', result)

    def test_decode_returns_string(self):
        audio  = _morse_audio('CQ', wpm=20, freq=800)
        engine = MorseEngine(SR, audio)
        self.assertIsInstance(engine.decode(), str)


class TestDecodeWavHelper(unittest.TestCase):

    def test_missing_file_returns_error_string(self):
        result = decode_wav('/nonexistent/path/audio.wav')
        self.assertIn('Error', result)

    def test_valid_wav_returns_string(self):
        wav_path = os.path.join(TEST_AUDIO, 'test_sos.wav')
        if not os.path.exists(wav_path):
            self.skipTest('test_sos.wav not found in data/test_audio/')
        result = decode_wav(wav_path)
        self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main()
