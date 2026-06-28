"""
Train the MorseDecoder model on the generated dataset.
Run scripts/generate_dataset.py first, then: python scripts/train.py
Saves the best checkpoint to model_best.pt.
"""

import json
import os
import sys

# Allow importing from src/ regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from src.model import BLANK_IDX, CHAR_TO_IDX, MorseDecoder, greedy_decode


class MorseDataset(Dataset):
    def __init__(self, meta_path: str, n_mels: int = 64, sr: int = 8000, max_samples: int = None):
        with open(meta_path) as f:
            self.meta = json.load(f)
        if max_samples is not None:
            self.meta = self.meta[:max_samples]
        self.audio_dir = os.path.join(os.path.dirname(os.path.abspath(meta_path)), 'audio')
        self.mel = T.MelSpectrogram(sample_rate=sr, n_fft=256, hop_length=64, n_mels=n_mels)
        self.db  = T.AmplitudeToDB()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        fpath = item['file']
        if not os.path.isabs(fpath):
            fpath = os.path.join(self.audio_dir, fpath)
        _, raw = wavfile.read(fpath)
        audio = torch.FloatTensor(raw.astype(np.float32) / 32768.0).unsqueeze(0)
        spec = self.db(self.mel(audio))   # (1, n_mels, T)

        label = torch.LongTensor(
            [CHAR_TO_IDX[c] for c in item['text'].upper() if c in CHAR_TO_IDX]
        )
        return spec, label


def collate(batch):
    """Pad spectrograms to the longest in the batch; flatten labels for CTC."""
    specs, labels = zip(*batch)
    T_max = max(s.shape[-1] for s in specs)
    B, n_mels = len(specs), specs[0].shape[1]

    padded = torch.zeros(B, 1, n_mels, T_max)
    input_lens = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(specs):
        t = s.shape[-1]
        padded[i, 0, :, :t] = s
        input_lens[i] = t

    target_lens = torch.LongTensor([len(l) for l in labels])
    targets = torch.cat(labels)   # CTC expects a 1-D concatenated target tensor
    return padded, input_lens, targets, target_lens


def train(
    dataset_dir: str = 'data/synthetic',
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    save_path: str = 'model_best.pt',
    max_samples: int = 1000,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    full = MorseDataset(os.path.join(dataset_dir, 'metadata.json'), max_samples=max_samples)
    n_val   = max(1, int(len(full) * 0.1))
    n_train = len(full) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f'Train: {n_train}  Val: {n_val}')

    train_dl = DataLoader(train_set, batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(val_set,   batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0)

    model = MorseDecoder().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    ctc   = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)

    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        t_loss = 0.0
        train_iter = tqdm(train_dl, desc=f'Epoch {epoch:3d}/{epochs} [train]',
                          leave=False) if _TQDM else train_dl
        for specs, in_lens, tgts, tgt_lens in train_iter:
            specs = specs.to(device)
            tgts  = tgts.to(device)

            log_probs = model(specs).permute(1, 0, 2)   # (T, B, C) for CTCLoss
            loss = ctc(log_probs, tgts, in_lens, tgt_lens)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            t_loss += loss.item()
            if _TQDM:
                train_iter.set_postfix(loss=f'{loss.item():.4f}')

        # --- Validation ---
        model.eval()
        v_loss = 0.0
        val_iter = tqdm(val_dl, desc=f'Epoch {epoch:3d}/{epochs} [val]  ',
                        leave=False) if _TQDM else val_dl
        with torch.no_grad():
            for specs, in_lens, tgts, tgt_lens in val_iter:
                specs = specs.to(device)
                tgts  = tgts.to(device)
                lp = model(specs).permute(1, 0, 2)
                v_loss += ctc(lp, tgts, in_lens, tgt_lens).item()

        avg_t = t_loss / len(train_dl)
        avg_v = v_loss / len(val_dl)
        sched.step(avg_v)

        tag = ''
        if avg_v < best_val:
            best_val = avg_v
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': best_val}, save_path)
            tag = ' ← saved'

        print(f'Epoch {epoch:3d}/{epochs}  train={avg_t:.4f}  val={avg_v:.4f}{tag}')

    print(f'\nTraining complete. Best val loss: {best_val:.4f}')
    print(f'Model saved to: {save_path}')
    print('Next step: python inference.py <audio.wav>')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Train the MorseDecoder model')
    ap.add_argument('--dataset',    default='data/synthetic',
                    help='Dataset directory containing metadata.json (default: data/synthetic)')
    ap.add_argument('--epochs',     type=int,   default=15)
    ap.add_argument('--batch-size', type=int,   default=32)
    ap.add_argument('--lr',         type=float, default=3e-4)
    ap.add_argument('--save',        default='model_best.pt',
                    help='Path to save best checkpoint (default: model_best.pt)')
    ap.add_argument('--max-samples', type=int, default=1000,
                    help='Use only the first N samples from metadata.json (default: 1000)')
    args = ap.parse_args()
    train(dataset_dir=args.dataset, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, save_path=args.save,
          max_samples=args.max_samples)
