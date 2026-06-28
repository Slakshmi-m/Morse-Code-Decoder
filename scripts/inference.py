"""
Decode a WAV file using the trained MorseDecoder model.
Usage: python scripts/inference.py <audio.wav> [model_best.pt]
Requires model_best.pt produced by scripts/train.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchaudio.transforms as T
from scipy.io import wavfile

from src.model import MorseDecoder, greedy_decode


def decode_wav_ml(file_path: str, model_path: str = 'model_best.pt',
                  n_mels: int = 64, sr: int = 8000) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(model_path, map_location=device)
    model = MorseDecoder(n_mels=n_mels).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    _, raw = wavfile.read(file_path)
    audio = torch.FloatTensor(raw.astype(np.float32) / 32768.0).unsqueeze(0)

    mel = T.MelSpectrogram(sample_rate=sr, n_fft=256, hop_length=64, n_mels=n_mels)
    spec = T.AmplitudeToDB()(mel(audio))      # (1, n_mels, T)
    spec = spec.unsqueeze(0).to(device)       # (1, 1, n_mels, T)

    with torch.no_grad():
        log_probs = model(spec)[0]            # (T, VOCAB_SIZE)

    return greedy_decode(log_probs)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python inference.py <audio.wav> [model_best.pt]')
        sys.exit(1)
    wav = sys.argv[1]
    mdl = sys.argv[2] if len(sys.argv) > 2 else 'model_best.pt'
    print(f'Decoded: {decode_wav_ml(wav, mdl)}')
