"""
model.py — CRNN Morse Decoder (CNN + LSTM)
==========================================
Architecture:
  Conv1d x3  — extract local tone/silence features from mel spectrogram
  LSTM x2    — model the sequence (remembers dots/dashes across time)
  Linear     — project to character probabilities for CTC

Input  : (B, 1, n_mels, T)  log mel spectrogram
Output : (B, T, VOCAB_SIZE)  per-frame log-probabilities
"""

from __future__ import annotations

import torch
import torch.nn as nn

VOCAB: list[str]             = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
VOCAB_SIZE: int              = len(VOCAB) + 1
BLANK_IDX: int               = 0
CHAR_TO_IDX: dict[str, int]  = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX_TO_CHAR: dict[int, str]  = {i + 1: c for i, c in enumerate(VOCAB)}


def greedy_decode(log_probs: torch.Tensor) -> str:
    """Collapse CTC per-frame log-probabilities into a text string."""
    indices = log_probs.argmax(dim=-1).tolist()
    chars: list[str] = []
    prev = None
    for idx in indices:
        if idx != prev and idx != BLANK_IDX:
            chars.append(IDX_TO_CHAR.get(idx, ""))
        prev = idx
    return "".join(chars).strip()


class MorseDecoder(nn.Module):
    """
    CRNN Morse decoder — Conv1d extracts local features, LSTM models the sequence.
    """

    def __init__(self, n_mels: int = 32, cnn_channels: int = 64,
                 hidden: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_channels, hidden, layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.2 if layers > 1 else 0.0)
        self.drop = nn.Dropout(0.1)
        self.fc   = nn.Linear(hidden * 2, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)           # (B, n_mels, T)
        x = self.conv(x)           # (B, cnn_channels, T)
        x = x.permute(0, 2, 1)    # (B, T, cnn_channels)
        x, _ = self.lstm(x)        # (B, T, hidden*2)
        return self.fc(self.drop(x)).log_softmax(dim=-1)


MorseDecoderCNN = MorseDecoder  # backward-compat alias
