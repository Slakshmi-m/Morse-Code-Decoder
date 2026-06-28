"""
CNN-LSTM model for end-to-end Morse code decoding using CTC loss.
"""

import torch
import torch.nn as nn

# All decodable characters (must match generate_dataset.py MORSE_TABLE keys + space)
VOCAB = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?-/= ')
VOCAB_SIZE = len(VOCAB) + 1   # +1 for CTC blank token
BLANK_IDX = len(VOCAB)

CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}


def greedy_decode(log_probs: torch.Tensor) -> str:
    """CTC greedy decode: argmax then collapse repeated tokens and blanks."""
    best = log_probs.argmax(dim=-1).tolist()
    out, prev = [], BLANK_IDX
    for idx in best:
        if idx != BLANK_IDX and idx != prev:
            out.append(IDX_TO_CHAR.get(idx, ''))
        prev = idx
    return ''.join(out)


class MorseDecoder(nn.Module):
    """
    CNN feature extractor + bidirectional LSTM + CTC output head.

    Input:  (B, 1, n_mels, T) — log mel spectrogram
    Output: (B, T, VOCAB_SIZE) — log softmax probabilities per time frame
    """

    def __init__(self, n_mels: int = 64, hidden: int = 256, layers: int = 2):
        super().__init__()

        # Three conv blocks, each halving the frequency dimension only
        # (MaxPool2d kernel=(2,1) pools freq, leaves time unchanged)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        # After 3x freq-halving: freq = n_mels // 8
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
        self.fc = nn.Linear(hidden * 2, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                                     # (B, 128, F//8, T)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)     # (B, T, features)
        x, _ = self.rnn(self.drop(x))                       # (B, T, hidden*2)
        return self.fc(self.drop(x)).log_softmax(dim=-1)    # (B, T, VOCAB_SIZE)
