"""
model.py — Simple MLP Morse Decoder (ML Path)
==============================================
A beginner-friendly replacement for the CNN-LSTM model.

Architecture
------------
  Input    (B, 1, n_mels, T)  — log mel spectrogram
  Reshape  (B*T, n_mels)      — treat every time frame independently
  MLP      three Linear layers with ReLU activations
  Output   (B, T, VOCAB_SIZE) — per-frame character probabilities

How it works (plain English)
-----------------------------
A mel spectrogram turns audio into a 2-D image:
  - rows   = frequency buckets (n_mels of them)
  - columns = time frames (T of them)

The CNN-LSTM would scan across both dimensions at once.
This MLP is simpler: it looks at ONE column (one time frame) at a time,
runs it through three layers of weighted sums + ReLU, and outputs
a probability for each possible character at that moment.
CTC loss (in train.py) then figures out how to align those per-frame
probabilities into the final decoded text — so we never need to tell
the model exactly which frame corresponds to which letter.

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
    CTC greedy decode: argmax at each time step → collapse repeats → remove blanks.

    Example:
        frame probs: [A, A, blank, B, B, B, blank, blank, C]
        collapsed  : [A, blank, B, blank, blank, C]
        no-blank   : "ABC"

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
    Three-layer MLP Morse Decoder — processes each mel frame independently.

    Why an MLP?
    -----------
    A Multi-Layer Perceptron (MLP) is the simplest kind of neural network.
    Each layer is just:  output = ReLU( W * input + b )
    where W is a matrix of learnable weights and b is a bias vector.
    Three of these stacked = "deep" enough to learn useful patterns,
    simple enough to see exactly what's happening at each step.

    Parameters
    ----------
    n_mels  : int   Number of mel frequency bins (must match train.py). Default 64.
    hidden  : int   Width of the hidden layers. Default 256.
    """

    def __init__(self, n_mels: int = 64, hidden: int = 256) -> None:
        super().__init__()

        # Layer 1: n_mels → hidden  (e.g. 64 → 256)
        # Layer 2: hidden → hidden//2  (e.g. 256 → 128)
        # Layer 3: hidden//2 → VOCAB_SIZE  (e.g. 128 → 45)
        self.mlp = nn.Sequential(
            nn.Linear(n_mels,      hidden),
            nn.ReLU(),
            nn.Dropout(0.3),          # randomly zero 30% of neurons → prevents memorising training data
            nn.Linear(hidden,      hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, VOCAB_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 1, n_mels, T)
            B = batch size, 1 = single audio channel,
            n_mels = frequency bins, T = time frames

        Returns
        -------
        torch.Tensor  shape (B, T, VOCAB_SIZE)
            Log-probability of each character at each time frame.
        """
        b, _, n_mels, t = x.shape

        # Step 1 — remove the channel dimension and move time to last
        #   (B, 1, n_mels, T) → (B, n_mels, T) → (B, T, n_mels)
        x = x.squeeze(1).permute(0, 2, 1)

        # Step 2 — merge batch and time so the MLP sees one frame at a time
        #   (B, T, n_mels) → (B*T, n_mels)
        x = x.reshape(b * t, n_mels)

        # Step 3 — run MLP on every frame in one shot
        #   (B*T, n_mels) → (B*T, VOCAB_SIZE)
        x = self.mlp(x)

        # Step 4 — restore batch and time dimensions
        #   (B*T, VOCAB_SIZE) → (B, T, VOCAB_SIZE)
        x = x.reshape(b, t, -1)

        # log_softmax converts raw scores → log-probabilities (required by CTC loss)
        return x.log_softmax(dim=-1)
