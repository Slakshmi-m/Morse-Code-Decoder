import os, shutil, subprocess
import numpy as np
from scipy.io import wavfile


def find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if os.path.isfile(p):
            return p
    return None


_FFMPEG = find_ffmpeg()


def load_audio_file(path: str):
    """Return (sample_rate, int16_mono_array) for WAV or MP3."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        if not _FFMPEG:
            raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
        result = subprocess.run(
            [_FFMPEG, "-i", path, "-f", "s16le", "-ac", "1",
             "-ar", "44100", "-loglevel", "error", "-"],
            capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[-300:]}")
        return 44100, np.frombuffer(result.stdout, dtype=np.int16).copy()
    fs, raw = wavfile.read(path)
    return fs, raw
