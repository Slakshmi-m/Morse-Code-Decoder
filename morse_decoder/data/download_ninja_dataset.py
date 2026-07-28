"""
download_ninja_dataset.py — Morse Code Ninja Dataset Downloader & Processor
=============================================================================
Downloads real Morse Code Ninja practice audio (MP3/M4A), extracts the
Morse-only portion (before the spoken answer), converts to WAV at 8 kHz,
and writes a metadata.json compatible with train.py.

HOW THE NINJA FILES WORK
-------------------------
Each file is named like:  "E - dit.mp3"  or  "THE - 15wpm.mp3"
Structure of the audio:
  [silence] [Morse code tones] [pause] [spoken answer] [silence]

We keep only the Morse-code portion by detecting where the high-frequency
tone energy drops off before the voice starts (voices have most energy
below 400 Hz; Morse tones are typically 600-800 Hz).

WHAT IS DOWNLOADED (manageable size for a student project)
----------------------------------------------------------
  1. Single Letters  — 15 wpm and 20 wpm   (letters A-Z one at a time)
  2. Single Letter-Number — 15 wpm          (A-Z + 0-9)
  3. Top 100 Words Farnsworth 3x spacing — 15 wpm and 20 wpm
  4. Top 200 Words Farnsworth 3x spacing — 15 wpm

Total estimated download: ~80-150 MB compressed.
Uncompressed WAVs (8 kHz mono 16-bit): ~200-400 MB.

REQUIREMENTS
------------
    pip install requests pydub scipy numpy
    # Also needs ffmpeg for MP3 decoding:
    #   Windows: winget install ffmpeg
    #   macOS:   brew install ffmpeg
    #   Linux:   sudo apt install ffmpeg

USAGE
-----
    python download_ninja_dataset.py              # download + process
    python download_ninja_dataset.py --skip-download  # reprocess existing ZIPs

OUTPUT
------
    ninja_dataset/
        zips/     raw downloaded ZIP files (kept so you can re-run --skip-download)
        audio/    extracted 8 kHz mono WAV files
        metadata.json  compatible with train.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
import requests
from scipy.io import wavfile
from scipy.signal import butter, lfilter

# ─────────────────────────────────────────────────────────────────────────────
# Morse Code Ninja download URLs
# Base CDN: https://podcasts-morsecode-ninja.nyc3.cdn.digitaloceanspaces.com
# ─────────────────────────────────────────────────────────────────────────────
CDN = "https://podcasts-morsecode-ninja.nyc3.cdn.digitaloceanspaces.com/practice/files/individual-files"

# (label, filename_template, wpm_list)
# The filename uses URL encoding for spaces
NINJA_SETS = [
    # Single letters — best for character-level training
    ("Single Letters",         "Single Letters {wpm}wpm.zip",          [15, 20, 25]),
    ("Single Letter-Number",   "Single Letter-Number {wpm}wpm.zip",    [15, 20]),
    # Farnsworth word sets — real sending style / spacing
    ("Top100 Words 3x Fw",     "Top 100 Words - 3x Character Spacing - {wpm}wpm.zip", [15, 20]),
    ("Top200 Words 3x Fw",     "Top 200 Words - 3x Character Spacing - {wpm}wpm.zip", [15]),
    ("Top100 Words 4x Fw",     "Top 100 Words - 4x Character Spacing - {wpm}wpm.zip", [15]),
]

TARGET_SR    = 8_000   # Hz — must match train.py / engine.py
TONE_LOW_HZ  = 400     # Morse tones are above this
VOICE_LOW_HZ = 80      # Human voice fundamental starts here
VOICE_HIGH_HZ= 350     # Voice energy mostly below this

# ─────────────────────────────────────────────────────────────────────────────
# Label extraction from filename
# ─────────────────────────────────────────────────────────────────────────────

# Morse ↔ text table (for label lookup)
MORSE_TO_CHAR = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    ".----": "1", "..---": "2", "...--": "3", "....-": "4", ".....": "5",
    "-....": "6", "--...": "7", "---..": "8", "----.": "9", "-----": "0",
}

# Common Ninja filename patterns:
#   "E - dit.mp3"           → label "E"
#   "THE - 15wpm.mp3"       → label "THE"
#   "A 15wpm.mp3"           → label "A"
#   "73 15wpm.mp3"          → label "73"
_LABEL_RE = re.compile(
    r"^([A-Z0-9 /]+?)"        # capture the text part
    r"(?:\s*[-–]\s*|\s+)"     # separator: " - " or just whitespace
    r"(?:\d+wpm|dit|dah|.*)",  # remainder (speed or phonetic)
    re.IGNORECASE,
)


def extract_label(filename: str) -> Optional[str]:
    """
    Guess the Morse text label from the MP3 filename.
    Returns the label in upper-case, or None if unrecognisable.
    """
    stem = Path(filename).stem.strip()
    m    = _LABEL_RE.match(stem)
    if m:
        label = m.group(1).strip().upper()
        # Sanity: must only contain A-Z 0-9 space
        if re.fullmatch(r"[A-Z0-9 ]+", label) and label:
            return label
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Audio processing
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(data: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    nyq  = sr / 2
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return lfilter(b, a, data.astype(np.float32))


def _rms_envelope(data: np.ndarray, window_samples: int) -> np.ndarray:
    """Short-time RMS energy."""
    sq   = data.astype(np.float64) ** 2
    kern = np.ones(window_samples) / window_samples
    return np.sqrt(np.convolve(sq, kern, mode="same"))


def trim_to_morse(audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    Remove the spoken-answer portion from a Ninja audio file.

    Strategy
    --------
    1. Compute short-time RMS of the Morse-band (400-1200 Hz).
    2. Compute short-time RMS of the voice-band  (80-350 Hz).
    3. Find the point where voice energy exceeds tone energy by >6 dB
       — that is where speaking starts.  Trim there.
    4. If no voice detected (some files are Morse-only), return as-is.

    Returns None if the audio is too short to be usable.
    """
    MIN_DURATION = 0.1   # seconds
    if len(audio) < int(sr * MIN_DURATION):
        return None

    win  = int(sr * 0.02)   # 20 ms window
    tone  = _rms_envelope(_bandpass(audio, sr, 400, 1200), win)
    voice = _rms_envelope(_bandpass(audio, sr, 80,  350),  win)

    # Find where voice power exceeds tone power (voice kicked in)
    ratio = voice / (tone + 1e-9)
    voice_start_samples = None
    for i in range(len(ratio) - win):
        if np.mean(ratio[i:i + win]) > 2.0:   # voice > 2× tone energy
            voice_start_samples = i
            break

    if voice_start_samples is None:
        trimmed = audio   # no voice detected — whole clip is Morse
    else:
        # Give a 50 ms buffer before voice onset
        cut = max(0, voice_start_samples - int(sr * 0.05))
        trimmed = audio[:cut]

    if len(trimmed) < int(sr * MIN_DURATION):
        return None

    return trimmed


def mp3_bytes_to_wav_array(mp3_bytes: bytes, target_sr: int) -> Optional[np.ndarray]:
    """
    Decode MP3 bytes → numpy int16 array at target_sr using pydub + ffmpeg.
    Returns None on failure.
    """
    try:
        from pydub import AudioSegment
        seg  = AudioSegment.from_file(io.BytesIO(mp3_bytes))
        seg  = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
        return np.frombuffer(seg.raw_data, dtype=np.int16)
    except Exception as exc:
        print(f"    [mp3 decode error] {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_zip(url: str, dest: Path) -> bool:
    """Download a ZIP to dest, return True on success."""
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [cache] {dest.name}")
        return True
    try:
        print(f"  [download] {dest.name} …", end=" ", flush=True)
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        print(f"{dest.stat().st_size // 1024} KB")
        return True
    except Exception as exc:
        print(f"FAILED — {exc}")
        if dest.exists():
            dest.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(output_dir: str = "ninja_dataset",
                  skip_download: bool = False) -> None:

    out      = Path(output_dir)
    zip_dir  = out / "zips"
    audio_dir = out / "audio"
    zip_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict] = []
    file_idx = 0
    total_skipped = 0

    for set_label, name_tmpl, wpm_list in NINJA_SETS:
        for wpm in wpm_list:
            filename = name_tmpl.format(wpm=wpm)
            url      = f"{CDN}/{quote(filename)}"
            dest_zip = zip_dir / filename

            print(f"\n── {set_label} @ {wpm} wpm ──")

            if not skip_download:
                ok = download_zip(url, dest_zip)
                if not ok:
                    continue
            elif not dest_zip.exists():
                print(f"  [skip] ZIP not found locally: {dest_zip}")
                continue

            # Process each MP3 inside the ZIP
            try:
                with zipfile.ZipFile(dest_zip) as zf:
                    mp3_names = [n for n in zf.namelist()
                                 if n.lower().endswith((".mp3", ".m4a"))]
                    print(f"  {len(mp3_names)} audio files in ZIP")

                    for mp3_name in mp3_names:
                        label = extract_label(os.path.basename(mp3_name))
                        if label is None:
                            total_skipped += 1
                            continue

                        # Decode MP3
                        raw_bytes = zf.read(mp3_name)
                        arr = mp3_bytes_to_wav_array(raw_bytes, TARGET_SR)
                        if arr is None:
                            total_skipped += 1
                            continue

                        # Trim spoken answer
                        trimmed = trim_to_morse(arr, TARGET_SR)
                        if trimmed is None:
                            total_skipped += 1
                            continue

                        # Save WAV
                        out_fname = f"ninja_{file_idx:06d}.wav"
                        wavfile.write(audio_dir / out_fname, TARGET_SR, trimmed)

                        metadata.append({
                            "file":        out_fname,
                            "text":        label,
                            "wpm":         wpm,
                            "source":      "morse_code_ninja",
                            "origin_file": os.path.basename(mp3_name),
                        })
                        file_idx += 1

            except zipfile.BadZipFile as exc:
                print(f"  [bad zip] {exc}")
                continue

    # Write metadata
    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*55}")
    print(f"Ninja dataset saved to: {output_dir}/")
    print(f"  Total WAV files : {file_idx}")
    print(f"  Skipped         : {total_skipped}")
    print(f"  metadata.json   : {meta_path}")
    print(f"\nNext step:  python train.py --dataset ninja_dataset")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download & process Morse Code Ninja audio for ML training."
    )
    parser.add_argument(
        "--output-dir", default="ninja_dataset",
        help="Where to save the processed dataset (default: ninja_dataset/)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading ZIPs — reprocess already-downloaded ZIPs only.",
    )
    args = parser.parse_args()
    build_dataset(output_dir=args.output_dir, skip_download=args.skip_download)
