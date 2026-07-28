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

import sys

import numpy as np
import torch
import torchaudio.transforms as T
from scipy.io import wavfile

from ml.model import MorseDecoder, greedy_decode


def decode_wav_ml(file_path: str, model_path: str = "model_best.pt",
                  n_mels: int = 64, sr: int = 8_000) -> str:
    """
    Decode a WAV file using the trained CNN-LSTM model.

    Parameters
    ----------
    file_path  : Path to input WAV file.
    model_path : Path to checkpoint produced by train.py.
    n_mels     : Number of mel filterbank channels (must match training).
    sr         : Expected sample rate in Hz.

    Returns
    -------
    str  Decoded text, or an error message.
    """
    if not torch.serialization:
        return "[PyTorch not available]"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        return f"[Checkpoint not found: {model_path}  — run train.py first]"

    model = MorseDecoder(n_mels=n_mels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    try:
        _, raw = wavfile.read(file_path)
    except Exception as exc:
        return f"[File error: {exc}]"

    audio = torch.FloatTensor(raw.astype(np.float32) / 32768.0).unsqueeze(0)
    mel   = T.MelSpectrogram(sample_rate=sr, n_fft=256, hop_length=64, n_mels=n_mels)
    spec  = T.AmplitudeToDB()(mel(audio))   # (1, n_mels, T)
    spec  = spec.unsqueeze(0).to(device)    # (1, 1, n_mels, T)

    with torch.no_grad():
        log_probs = model(spec)[0]          # (T, VOCAB_SIZE)

    return greedy_decode(log_probs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio.wav> [model_best.pt]")
        sys.exit(1)
    wav = sys.argv[1]
    mdl = sys.argv[2] if len(sys.argv) > 2 else "model_best.pt"
    print(f"Decoded: {decode_wav_ml(wav, mdl)}")
