"""
model_cnn.py — CNN-LSTM Morse Decoder (Advanced ML Path)
=========================================================
Architecture
------------
  Input   (B, 1, n_mels, T)  — log mel spectrogram
  CNN     three conv blocks, each halving the mel-frequency dimension
  LSTM    bidirectional 2-layer LSTM over the time axis
  Linear  projection to VOCAB_SIZE log-softmax probabilities
  Output  (B, T, VOCAB_SIZE) — per-frame character probabilities

How it works
------------
CNN scans the spectrogram like a 2D image — finds local patterns
(a bright stripe at 700 Hz = a tone burst = a dit or dah).
LSTM then reads the CNN features left-to-right (and right-to-left
in the bidirectional pass) keeping memory of what came before,
so it can learn sequences: dit → gap → dah = "A".

Slower to train than the MLP but more powerful for sequences.

Training: train.py --model cnn
Inference: inference.py --model cnn
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.model import BLANK_IDX, CHAR_TO_IDX, IDX_TO_CHAR, VOCAB, VOCAB_SIZE, greedy_decode

__all__ = ["MorseDecoderCNN"]


class MorseDecoderCNN(nn.Module):
    """
    CNN feature extractor + bidirectional LSTM + CTC output head.

    Parameters
    ----------
    n_mels  : int   Number of mel filterbank channels (default 64).
    hidden  : int   LSTM hidden size per direction (default 256).
    layers  : int   Number of LSTM layers (default 2).
    """

    def __init__(self, n_mels: int = 64, hidden: int = 256,
                 layers: int = 2) -> None:
        super().__init__()

        # Three conv blocks — each MaxPool2d(2,1) halves the mel dimension,
        # leaving the time axis intact so CTC can align over the full sequence.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        # After 3× mel-halving: effective mel dim = n_mels // 8
        rnn_input = (n_mels // 8) * 128

        self.rnn = nn.LSTM(
            input_size=rnn_input,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.3 if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hidden * 2, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 1, n_mels, T)

        Returns
        -------
        torch.Tensor  shape (B, T, VOCAB_SIZE)  — log-softmax probabilities
        """
        x = self.cnn(x)                                    # (B, 128, n_mels//8, T)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)    # (B, T, features)
        x, _ = self.rnn(self.drop(x))                      # (B, T, hidden*2)
        return self.fc(self.drop(x)).log_softmax(dim=-1)   # (B, T, VOCAB_SIZE)
