"""
train.py — Train the MorseDecoder CNN-LSTM Model
=================================================
Loads pre-generated WAV files from dataset/ — fast training.

Run generate_dataset.py first, then:
    python ml/train.py

Saves the best checkpoint (lowest val CTC loss) to model_best.pt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset

from ml.model import BLANK_IDX, CHAR_TO_IDX, MorseDecoderCNN as MorseDecoder, greedy_decode

# ── Morse dictionary for quick-decode test audio synthesis ───────────────────
MORSE_DICT: dict[str, str] = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',   'E': '.',
    'F': '..-.',  'G': '--.',   'H': '....',  'I': '..',    'J': '.---',
    'K': '-.-',   'L': '.-..',  'M': '--',    'N': '-.',    'O': '---',
    'P': '.--.',  'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',  'Y': '-.--',
    'Z': '--..',  '0': '-----', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.',
}


def _synth_morse(text: str, wpm: int = 22, fs: int = 8_000,
                 freq: float = 700.0) -> np.ndarray:
    """Render text as clean int16 Morse audio for the quick-decode test."""
    dot  = 1.2 / wpm
    segs = [np.zeros(int(fs * 0.3))]
    for ch in text.upper():
        if ch == ' ':
            segs.append(np.zeros(int(dot * 7 * fs)))
            continue
        code = MORSE_DICT.get(ch)
        if not code:
            continue
        for sym in code:
            dur = dot * 3 if sym == '-' else dot
            t   = np.arange(int(dur * fs)) / fs
            segs.append((np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32))
            segs.append(np.zeros(int(dot * fs)))
        segs.append(np.zeros(int(dot * 3 * fs)))
    segs.append(np.zeros(int(fs * 0.3)))
    audio = np.concatenate(segs)
    return (audio * 16384).clip(-32767, 32767).astype(np.int16)


def _quick_decode_test(model: MorseDecoder, device: torch.device,
                       n_mels: int = 32, sr: int = 8_000) -> None:
    """Run 8 known words through the model and print a score card."""
    TEST_WORDS = ["E", "A", "SOS", "CQ", "HAM", "AB", "OK", "TNX"]
    mel = T.MelSpectrogram(sample_rate=sr, n_fft=256, hop_length=64, n_mels=n_mels)
    db  = T.AmplitudeToDB()
    model.eval()
    correct = 0

    print("\n--- Quick decode test ---")
    with torch.no_grad():
        for word in TEST_WORDS:
            samples = _synth_morse(word, fs=sr)
            tensor  = torch.FloatTensor(samples.astype(np.float32) / 32768.0).unsqueeze(0)
            spec    = db(mel(tensor)).unsqueeze(0).to(device)
            result  = greedy_decode(model(spec)[0]).strip()
            ok      = result == word
            correct += int(ok)
            tag     = "[OK]   " if ok else "[WRONG]"
            print(f"  {tag}  expected={word:<6}  got={result!r}")

    print(f"\n  Score: {correct}/{len(TEST_WORDS)}")
    if correct == len(TEST_WORDS):
        print("  Perfect score — model is ready!")
    elif correct >= len(TEST_WORDS) // 2:
        print("  Model looks good — minor errors expected on edge cases.")
    else:
        print("  Model still not converged — retrain with more epochs.")


class MorseDataset(Dataset):
    """PyTorch dataset wrapping a metadata.json + audio/ directory."""

    def __init__(self, meta_path: str, n_mels: int = 32,
                 sr: int = 8_000) -> None:
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.audio_dir = os.path.join(
            os.path.dirname(os.path.abspath(meta_path)), "audio"
        )
        self.mel = T.MelSpectrogram(sample_rate=sr, n_fft=256,
                                    hop_length=64, n_mels=n_mels)
        self.db  = T.AmplitudeToDB()

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        item   = self.meta[idx]
        _, raw = wavfile.read(os.path.join(self.audio_dir, item["file"]))
        audio  = torch.FloatTensor(raw.astype(np.float32) / 32768.0).unsqueeze(0)
        spec   = self.db(self.mel(audio))   # (1, n_mels, T)

        label = torch.LongTensor(
            [CHAR_TO_IDX[c] for c in item["text"].upper() if c in CHAR_TO_IDX]
        )
        return spec, label


def collate(batch):
    specs, labels = zip(*batch)
    T_max  = max(s.shape[-1] for s in specs)
    n_mels = specs[0].shape[1]
    B      = len(specs)

    padded     = torch.zeros(B, 1, n_mels, T_max)
    input_lens = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(specs):
        t = s.shape[-1]
        padded[i, 0, :, :t] = s
        input_lens[i] = t

    target_lens = torch.LongTensor([len(l) for l in labels])
    targets     = torch.cat(labels)
    return padded, input_lens, targets, target_lens


def train(
    dataset_dir: str   = "dataset",
    epochs:      int   = 60,
    batch_size:  int   = 32,
    lr:          float = 3e-4,
    save_path:   str   = "model_best.pt",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    primary_meta = os.path.join(dataset_dir, "metadata.json")
    if not os.path.exists(primary_meta):
        print(f"ERROR: {primary_meta} not found.")
        print("Run:  python data/generate_dataset.py")
        return

    ds = MorseDataset(primary_meta)
    print(f"Loaded '{dataset_dir}': {len(ds)} samples")
    print(f"Total training samples: {len(ds)}")

    n_val   = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_set, val_set = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train}   Val: {n_val}")

    train_dl = DataLoader(train_set, batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(val_set,   batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0)

    N_MELS = 32
    model  = MorseDecoder(n_mels=N_MELS).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Model    : CRNN  params={total_p:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    ctc   = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    best_val = float("inf")

    n_train_batches = len(train_dl)
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for batch_idx, (specs, in_lens, tgts, tgt_lens) in enumerate(train_dl, 1):
            specs = specs.to(device)
            tgts  = tgts.to(device)
            log_probs = model(specs).permute(1, 0, 2)
            loss = ctc(log_probs, tgts, in_lens, tgt_lens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            t_loss += loss.item()
            if batch_idx % 20 == 0 or batch_idx == n_train_batches:
                print(f"  Epoch {epoch}/{epochs}  batch {batch_idx}/{n_train_batches}"
                      f"  loss={loss.item():.4f}", end="\r", flush=True)

        print()  # newline after \r progress
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for specs, in_lens, tgts, tgt_lens in val_dl:
                specs = specs.to(device)
                tgts  = tgts.to(device)
                lp = model(specs).permute(1, 0, 2)
                v_loss += ctc(lp, tgts, in_lens, tgt_lens).item()

        avg_t = t_loss / len(train_dl)
        avg_v = v_loss / len(val_dl)
        sched.step(avg_v)

        tag = ""
        if avg_v < best_val:
            best_val = avg_v
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    best_val,
                "n_mels":      N_MELS,
            }, save_path)
            tag = " ← saved"

        print(f"Epoch {epoch:3d}/{epochs}  train={avg_t:.4f}  val={avg_v:.4f}{tag}")

    print(f"\nTraining complete.  Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {save_path}")

    _quick_decode_test(model, device, n_mels=N_MELS)

    print("\nNext step: python main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="dataset")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--save",       default="model_best.pt")
    args = parser.parse_args()

    train(
        dataset_dir = args.dataset,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        save_path   = args.save,
    )
