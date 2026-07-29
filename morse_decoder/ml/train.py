"""
train.py — Train the MorseDecoder CNN-LSTM Model
=================================================
Works with EITHER:
  - Synthetic dataset:      python train.py
  - Morse Code Ninja data:  python train.py --dataset ninja_dataset
  - Both combined:          python train.py --dataset ninja_dataset --also-synthetic

Run generate_dataset.py or download_ninja_dataset.py first.
Saves the best checkpoint (lowest val CTC loss) to model_best.pt.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from scipy.io import wavfile
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ml.model import BLANK_IDX, CHAR_TO_IDX, MorseDecoder, greedy_decode


class MorseDataset(Dataset):
    """PyTorch dataset wrapping a metadata.json + audio/ directory."""

    def __init__(self, meta_path: str, n_mels: int = 64,
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
        item  = self.meta[idx]
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
    dataset_dir:     str   = "dataset",
    also_synthetic:  bool  = False,
    epochs:          int   = 60,
    batch_size:      int   = 32,
    lr:              float = 3e-4,
    save_path:       str   = "model_best.pt",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset(s) ───────────────────────────────────────────────────────
    datasets = []

    primary_meta = os.path.join(dataset_dir, "metadata.json")
    if os.path.exists(primary_meta):
        ds = MorseDataset(primary_meta)
        datasets.append(ds)
        print(f"Loaded '{dataset_dir}': {len(ds)} samples")
    else:
        print(f"ERROR: {primary_meta} not found.")
        print("Run:  python -m data.download_ninja_dataset   (Morse Code Ninja)")
        print("  or: python -m data.generate_dataset         (synthetic)")
        return

    if also_synthetic:
        synth_meta = os.path.join("dataset", "metadata.json")
        if os.path.exists(synth_meta):
            synth = MorseDataset(synth_meta)
            datasets.append(synth)
            print(f"Also loaded synthetic 'dataset': {len(synth)} samples")
        else:
            print("Warning: --also-synthetic requested but dataset/metadata.json not found.")
            print("Run python generate_dataset.py to create it.")

    full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"Total training samples: {len(full)}")

    n_val    = max(1, int(len(full) * 0.1))
    n_train  = len(full) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train}   Val: {n_val}")

    train_dl = DataLoader(train_set, batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(val_set,   batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0)

    model = MorseDecoder().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    ctc   = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for specs, in_lens, tgts, tgt_lens in train_dl:
            specs = specs.to(device)
            tgts  = tgts.to(device)
            log_probs = model(specs).permute(1, 0, 2)
            loss = ctc(log_probs, tgts, in_lens, tgt_lens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            t_loss += loss.item()

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
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "val_loss": best_val}, save_path)
            tag = " ← saved"

        print(f"Epoch {epoch:3d}/{epochs}  train={avg_t:.4f}  val={avg_v:.4f}{tag}")

    print(f"\nTraining complete.  Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {save_path}")
    print("Next step: python inference.py <audio.wav>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset",
                        help="Dataset directory containing metadata.json (default: dataset/). "
                             "Use 'ninja_dataset' for Morse Code Ninja data.")
    parser.add_argument("--also-synthetic", action="store_true",
                        help="Also mix in the synthetic dataset/ alongside the Ninja data.")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--save",       default="model_best.pt")
    args = parser.parse_args()

    train(
        dataset_dir    = args.dataset,
        also_synthetic = args.also_synthetic,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        save_path      = args.save,
    )
