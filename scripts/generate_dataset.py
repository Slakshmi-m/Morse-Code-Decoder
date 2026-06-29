import sys
import os
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.constants import TEXT_TO_MORSE

_DEFAULT_SR = 8000
_AMPLITUDE  = 16384


def text_to_audio(text, wpm=15, freq=700, noise=0.0, sample_rate=_DEFAULT_SR):
    """
    Generate a synthetic Morse code audio buffer for the given text.

    Returns a numpy int16 array at `sample_rate` Hz.
    """
    unit_sec = 1.2 / wpm  # duration of one dit in seconds (PARIS standard)

    def _tone(duration_sec):
        n = max(1, int(sample_rate * duration_sec))
        t = np.linspace(0, duration_sec, n, endpoint=False)
        return np.sin(2 * np.pi * freq * t) * _AMPLITUDE

    def _silence(duration_sec):
        return np.zeros(max(1, int(sample_rate * duration_sec)))

    parts = []
    words = text.upper().split(' ')

    for w_idx, word in enumerate(words):
        for c_idx, char in enumerate(word):
            morse = TEXT_TO_MORSE.get(char, '')
            if not morse:
                continue
            for e_idx, symbol in enumerate(morse):
                parts.append(_tone(unit_sec) if symbol == '.' else _tone(unit_sec * 3))
                if e_idx < len(morse) - 1:
                    parts.append(_silence(unit_sec))       # inter-element gap
            if c_idx < len(word) - 1:
                parts.append(_silence(unit_sec * 3))       # inter-character gap
        if w_idx < len(words) - 1:
            parts.append(_silence(unit_sec * 7))           # inter-word gap

    if not parts:
        return np.zeros(sample_rate, dtype=np.int16)

    audio = np.concatenate(parts).astype(np.float32)

    if noise > 0.0:
        rng = np.random.default_rng(42)
        audio += (rng.standard_normal(len(audio)) * noise * _AMPLITUDE).astype(np.float32)

    return np.clip(audio, -32768, 32767).astype(np.int16)


def save_wav(path, text, wpm=15, freq=700, noise=0.0, sample_rate=_DEFAULT_SR):
    audio = text_to_audio(text, wpm=wpm, freq=freq, noise=noise, sample_rate=sample_rate)
    wavfile.write(path, sample_rate, audio)


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_audio')
    os.makedirs(out_dir, exist_ok=True)
    save_wav(os.path.join(out_dir, 'test_sos.wav'),   'SOS',   wpm=15, freq=700)
    save_wav(os.path.join(out_dir, 'test_hello.wav'), 'HELLO', wpm=15, freq=700)
    print("Generated test_sos.wav and test_hello.wav")
