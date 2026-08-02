"""
inference.py — Decode a WAV File Using the Trained ML Model
============================================================
Usage:
    python inference.py <audio.wav>
    python inference.py <audio.wav> model_best.pt

Requires model_best.pt produced by train.py.
Falls back gracefully if the checkpoint is missing.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchaudio.transforms as T
from scipy.io import wavfile

from ml.model import MorseDecoder, greedy_decode

_model_cache: dict = {}
_CHUNK_SECONDS = 8.0


def _load_model(model_path: str, device: torch.device):
    key = (model_path, str(device))
    if key not in _model_cache:
        ckpt         = torch.load(model_path, map_location=device, weights_only=False)
        n_mels       = ckpt.get("n_mels", 32)
        cnn_channels = ckpt.get("cnn_channels", 64)
        hidden       = ckpt.get("hidden", 128)
        layers       = ckpt.get("layers", 2)
        model = MorseDecoder(n_mels=n_mels, cnn_channels=cnn_channels,
                             hidden=hidden, layers=layers).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache[key] = (model, n_mels)
    return _model_cache[key]


def _run_model(samples: np.ndarray, sr: int, model, n_mels: int, device) -> str:
    audio = torch.FloatTensor(samples.astype(np.float32) / 32768.0).unsqueeze(0)
    mel   = T.MelSpectrogram(sample_rate=sr, n_fft=256, hop_length=64, n_mels=n_mels)
    spec  = T.AmplitudeToDB()(mel(audio)).unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(spec)[0]
    return greedy_decode(log_probs)


def _find_split(samples: np.ndarray, sr: int, target: int, search_radius: int) -> int:
    win  = max(1, sr // 20)
    step = max(1, win // 4)
    lo   = max(0, target - search_radius)
    hi   = min(len(samples) - win, target + search_radius)
    best_pos, best_energy = target, float("inf")
    for pos in range(lo, hi, step):
        e = float(np.sum(samples[pos: pos + win].astype(np.float32) ** 2))
        if e < best_energy:
            best_energy, best_pos = e, pos + win // 2
    return int(best_pos)


def _split_into_chunks(samples: np.ndarray, sr: int, max_chunk: int) -> list:
    if len(samples) <= max_chunk:
        return [samples]
    target        = max_chunk
    search_radius = max_chunk // 3    # wider search → more likely to land on silence
    split         = _find_split(samples, sr, target, search_radius)
    split         = max(sr // 2, min(split, len(samples) - sr // 2))
    return (_split_into_chunks(samples[:split], sr, max_chunk) +
            _split_into_chunks(samples[split:], sr, max_chunk))


def decode_buffer_ml(samples: np.ndarray, sr: int = 8_000,
                     model_path: str = "model_best.pt") -> str:
    """Decode a numpy int16 audio buffer using the trained CRNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, n_mels = _load_model(model_path, device)
    except FileNotFoundError:
        return f"[Checkpoint not found: {model_path} — run: python ml/train.py]"
    except Exception as exc:
        return f"[Model load error: {exc}]"

    max_chunk = int(_CHUNK_SECONDS * sr)
    if len(samples) <= max_chunk:
        return _run_model(samples, sr, model, n_mels, device)

    chunks = _split_into_chunks(samples, sr, max_chunk)
    parts  = [_run_model(c, sr, model, n_mels, device)
              for c in chunks if len(c) >= sr // 4]
    return " ".join(p for p in parts if p)


def decode_wav_ml(file_path: str, model_path: str = "model_best.pt",
                  sr: int = 8_000) -> str:
    """Decode a WAV file using the trained CRNN model.

    Handles any sample rate, stereo, int16/int32/float32 WAV files.
    """
    try:
        file_sr, raw = wavfile.read(file_path)

        # Stereo → mono
        if raw.ndim > 1:
            raw = raw[:, 0]

        # Normalise to float32 [-1, 1]
        if raw.dtype == np.int16:
            audio = raw.astype(np.float32) / 32768.0
        elif raw.dtype == np.int32:
            audio = raw.astype(np.float32) / 2_147_483_648.0
        else:
            audio = raw.astype(np.float32)

        # Resample to 8 000 Hz if the file uses a different rate
        if file_sr != sr:
            n     = int(len(audio) * sr / file_sr)
            audio = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(audio)),
                audio,
            ).astype(np.float32)

        # Back to int16 — decode_buffer_ml expects int16 samples
        samples = (audio * 32767).clip(-32767, 32767).astype(np.int16)
        return decode_buffer_ml(samples, sr=sr, model_path=model_path)

    except Exception as exc:
        return f"[File error: {exc}]"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio.wav> [model_best.pt]")
        sys.exit(1)
    wav = sys.argv[1]
    mdl = sys.argv[2] if len(sys.argv) > 2 else "model_best.pt"
    print(f"Decoded: {decode_wav_ml(wav, mdl)}")
