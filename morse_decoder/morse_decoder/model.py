"""
model.py — CNN-LSTM Morse Decoder (ML Path)
============================================
End-to-end model trained with CTC loss.

Architecture
------------
  Input   (B, 1, n_mels, T)  — log mel spectrogram
  CNN     three conv blocks, each halving the mel-frequency dimension
  LSTM    bidirectional 2-layer LSTM over the time axis
  Linear  projection to VOCAB_SIZE log-softmax probabilities
  Output  (B, T, VOCAB_SIZE) — per-frame character probabilities

Training: train.py
Inference: inference.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary — must match generate_dataset.py MORSE_TABLE keys + space
# ─────────────────────────────────────────────────────────────────────────────
VOCAB: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?-/= ")
VOCAB_SIZE: int  = len(VOCAB) + 1   # +1 for CTC blank token
BLANK_IDX:  int  = len(VOCAB)

CHAR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR: dict[int, str] = {i: c for i, c in enumerate(VOCAB)}


def greedy_decode(log_probs: torch.Tensor) -> str:
    """
    CTC greedy decode: argmax → collapse repeated tokens → remove blanks.

    Parameters
    ----------
    log_probs : torch.Tensor  shape (T, VOCAB_SIZE)

    Returns
    -------
    str  Decoded text string.
    """
    best = log_probs.argmax(dim=-1).tolist()
    out, prev = [], BLANK_IDX
    for idx in best:
        if idx != BLANK_IDX and idx != prev:
            out.append(IDX_TO_CHAR.get(idx, ""))
        prev = idx
    return "".join(out)


class MorseDecoder(nn.Module):
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

        # CNN: three blocks each with MaxPool2d(2,1) — halves mel dimension,
        # leaves the time axis intact so CTC can align over the full sequence.
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
