"""
Prepare real Morse code audio for training.

Accepts a directory of audio files where the filename encodes the label:
  the.mp3        → label "THE"
  cq_de.mp3      → label "CQ DE"   (underscore = space)
  sos_5wpm.mp3   → label "SOS"     (trailing speed tag is stripped)

Also supports a CSV manifest for files that aren't self-labeling:
  --manifest labels.csv   (columns: file, label)

Output: data/real/audio/*.wav + data/real/metadata.json
Format is identical to generate_dataset.py output, so train.py works unchanged.

────────────────────────────────────────────────────────────────────────────────
HOW TO GET MORSE CODE NINJA AUDIO
────────────────────────────────────────────────────────────────────────────────
1. Visit  https://morsecode.ninja/practice/downloads/
2. Download any free word/character pack (ZIP file)
3. Unzip to a local folder, e.g.:  ~/Downloads/ninja_words/
4. Run:
     python prepare_real_data.py ~/Downloads/ninja_words/

Other free sources
  • LCWO (learn.cw-soft.de) — generate and download practice MP3s
  • G4FON Koch Trainer (g4fon.net) — exports WAV files
  • Any directory of WAV/MP3 files whose filenames spell out the Morse text
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import json
import os
import re

import numpy as np
from scipy.io import wavfile

# ── optional: pydub for MP3/OGG/FLAC/M4A ────────────────────────────────────
try:
    from pydub import AudioSegment
    _PYDUB = True
except ImportError:
    _PYDUB = False

# ── optional: torchaudio for MP3 fallback ────────────────────────────────────
try:
    import torch
    import torchaudio
    import torchaudio.transforms as _TA_T
    _TORCHAUDIO = True
except ImportError:
    _TORCHAUDIO = False

TARGET_SR = 8000
SUPPORTED_EXT = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
VALID_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?-/= ')

# Speed-hint patterns at the end of a filename — stripped before labeling
_SPEED_RE = re.compile(r'[_\-]\d{1,3}wpm$', re.IGNORECASE)


# ── label extraction ──────────────────────────────────────────────────────────

def label_from_filename(fname: str) -> str:
    """
    Turn a filename into its Morse text label.

    Examples:
      the.mp3          → "THE"
      cq_de.mp3        → "CQ DE"
      73.mp3           → "73"
      hello_20wpm.mp3  → "HELLO"
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    base = _SPEED_RE.sub('', base)          # strip trailing speed tag
    text = base.replace('_', ' ').upper()
    text = ''.join(c for c in text if c in VALID_CHARS)
    return text.strip()


# ── audio loading ─────────────────────────────────────────────────────────────

def _load_wav(path: str):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.float32 or data.dtype == np.float64:
        data = (data * 32767).clip(-32767, 32767).astype(np.int16)
    elif data.dtype == np.int32:
        data = (data >> 16).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)
    return data, sr


def _load_mp3_pydub(path: str):
    seg = (AudioSegment.from_file(path)
           .set_frame_rate(TARGET_SR)
           .set_channels(1)
           .set_sample_width(2))
    data = np.frombuffer(seg.raw_data, dtype=np.int16).copy()
    return data, TARGET_SR


def _load_mp3_torchaudio(path: str):
    waveform, sr = torchaudio.load(path)
    mono = waveform.mean(dim=0)                     # stereo → mono
    if sr != TARGET_SR:
        resamp = _TA_T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        mono = resamp(mono.unsqueeze(0)).squeeze(0)
    data = (mono.numpy() * 32767).clip(-32767, 32767).astype(np.int16)
    return data, TARGET_SR


def load_audio(path: str):
    """Load any supported audio file → (int16 array at TARGET_SR, sample_rate)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        return _load_wav(path)
    if ext in ('.mp3', '.ogg', '.flac', '.m4a', '.aac'):
        if _PYDUB:
            return _load_mp3_pydub(path)
        if _TORCHAUDIO:
            return _load_mp3_torchaudio(path)
        raise RuntimeError(
            f"Cannot decode {ext} — install pydub + ffmpeg:\n"
            "  pip install pydub\n"
            "  macOS : brew install ffmpeg\n"
            "  Linux : sudo apt install ffmpeg"
        )
    raise ValueError(f"Unsupported format: {ext}")


def _resample_int16(data: np.ndarray, orig: int, target: int) -> np.ndarray:
    if orig == target:
        return data
    n = int(len(data) * target / orig)
    out = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(data)),
        data.astype(np.float32),
    )
    return out.astype(np.int16)


# ── main ──────────────────────────────────────────────────────────────────────

def prepare(source_dir: str, output_dir: str = 'data/real',
            manifest: str = None):
    audio_out = os.path.join(output_dir, 'audio')
    os.makedirs(audio_out, exist_ok=True)

    # ── collect (path, label) pairs ───────────────────────────────────────────
    file_labels: list = []

    if manifest:
        with open(manifest, newline='') as f:
            for row in csv.DictReader(f):
                path = os.path.join(source_dir, row['file'])
                label = row['label'].upper().strip()
                if os.path.exists(path) and label:
                    file_labels.append((path, label))
    else:
        for root, _, files in os.walk(source_dir):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in SUPPORTED_EXT:
                    label = label_from_filename(fname)
                    if label:
                        file_labels.append((os.path.join(root, fname), label))

    print(f"Found {len(file_labels)} audio files in '{source_dir}'")
    if not file_labels:
        print("No files found — check your source directory.")
        return

    # ── process each file ─────────────────────────────────────────────────────
    metadata = []
    skipped = 0

    for i, (src_path, label) in enumerate(file_labels):
        try:
            data, sr = load_audio(src_path)
        except Exception as exc:
            print(f"  [skip] {os.path.basename(src_path)}: {exc}")
            skipped += 1
            continue

        if sr != TARGET_SR:
            data = _resample_int16(data, sr, TARGET_SR)

        out_name = f'real_{i:06d}.wav'
        wavfile.write(os.path.join(audio_out, out_name), TARGET_SR, data)

        metadata.append({
            'file': out_name,
            'text': label,
            'wpm': 0,       # unknown for real audio
            'freq': 0.0,
            'noise': -1.0,  # sentinel: this is real audio
            'fw_char_wpm': 0,
            'source': 'real',
        })

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(file_labels)}")

    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nReal dataset saved → '{output_dir}/'")
    print(f"  Processed : {len(metadata)}")
    print(f"  Skipped   : {skipped}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Prepare real Morse audio for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('source_dir',
                    help='Directory of MP3/WAV files (filename = Morse text)')
    ap.add_argument('--output-dir', default='data/real',
                    help='Output directory (default: data/real)')
    ap.add_argument('--manifest', default=None,
                    help='CSV with columns "file" and "label" for custom labeling')
    args = ap.parse_args()

    prepare(args.source_dir, args.output_dir, args.manifest)
