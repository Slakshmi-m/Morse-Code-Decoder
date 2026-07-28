"""
generate_dataset.py — Synthetic Morse Training Dataset Generator
=================================================================
Generates labelled audio samples covering:
  - All 42 Morse characters at 6 speeds × 2 frequencies (guaranteed coverage)
  - ~8 000 random samples with varied speed, frequency, noise, and
    Farnsworth spacing (as used by Morse Code Ninja)

Output layout:
    dataset/
        audio/sample_000000.wav  …
        metadata.json            [{file, text, wpm, freq, noise, fw_char_wpm}]

Run:
    python generate_dataset.py
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
from scipy.io import wavfile

from constants import MORSE_TABLE, ALL_CHARS

SAMPLE_RATE = 8_000

# Common words + ham radio Q-codes for realistic multi-character training
WORDS: list[str] = [
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
    "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID",
    "SOS", "CQ",  "DE",  "PARIS", "MORSE", "CODE", "NINJA", "TEST", "HELLO",
    "WORLD", "73", "88", "CW", "TNX", "AR", "SK", "QRN", "QRM", "HAM",
    "RADIO", "CALL", "SIGN", "FREQ", "BAND", "METER", "OM", "YL", "ES",
]


def text_to_audio(text: str, wpm: int, freq: float,
                  noise: float, fw_char_wpm: int = 0) -> np.ndarray:
    """
    Synthesise Morse code audio for the given text.

    Parameters
    ----------
    text         : Text to encode (upper-case, only chars in MORSE_TABLE).
    wpm          : Overall sending speed in words-per-minute.
    freq         : Carrier tone frequency in Hz.
    noise        : Noise amplitude as fraction of signal amplitude (0 = clean).
    fw_char_wpm  : Farnsworth character speed.  If > wpm, characters are sent
                   faster while letter/word gaps are stretched to hit the
                   slower overall wpm — mimicking Morse Code Ninja audio.
    """
    char_wpm = fw_char_wpm if fw_char_wpm > wpm else wpm
    dot   = 1.2 / char_wpm   # dot duration in seconds
    intra = dot               # intra-character symbol gap

    if char_wpm > wpm:
        # Farnsworth: PARIS = 50 units; distribute extra time into gaps.
        extra = max(0.0, (60.0 / wpm - 50 * dot) / 19)
        letter_gap = dot * 3 + extra * 3
        word_gap   = dot * 7 + extra * 7
    else:
        letter_gap = dot * 3
        word_gap   = dot * 7

    audio: list[float] = []

    def tone(d: float) -> None:
        n = int(SAMPLE_RATE * d)
        t = np.linspace(0, d, n, endpoint=False)
        audio.extend((np.sin(2 * np.pi * freq * t) * 16384).tolist())

    def silence(d: float) -> None:
        audio.extend([0.0] * int(SAMPLE_RATE * d))

    words = text.upper().split(" ")
    for wi, word in enumerate(words):
        chars = [c for c in word if c in MORSE_TABLE]
        for ci, ch in enumerate(chars):
            for si, sym in enumerate(MORSE_TABLE[ch]):
                tone(dot if sym == "." else dot * 3)
                if si < len(MORSE_TABLE[ch]) - 1:
                    silence(intra)
            if ci < len(chars) - 1:
                silence(letter_gap)
        if wi < len(words) - 1:
            silence(word_gap)

    if not audio:
        return np.zeros(SAMPLE_RATE, dtype=np.int16)

    sig = np.array(audio)
    sig = sig + np.random.normal(0, noise * 16384, sig.shape)
    return np.clip(sig, -32767, 32767).astype(np.int16)


def generate_dataset(output_dir: str = "dataset", n_random: int = 8_000) -> None:
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    samples: list[dict] = []

    # ── Guaranteed coverage: every char × 6 WPM × 2 freq ────────────────────
    for char in ALL_CHARS:
        for wpm in [5, 10, 15, 20, 25, 35]:
            for freq in [600, 800]:
                samples.append(dict(
                    text=char, wpm=wpm, freq=float(freq),
                    noise=random.uniform(0.0, 0.12), fw=0,
                ))

    # ── Random samples ────────────────────────────────────────────────────────
    for _ in range(n_random):
        wpm  = random.randint(5, 35)
        roll = random.random()

        if roll < 0.20:
            text = random.choice(ALL_CHARS)
        elif roll < 0.40:
            text = "".join(random.choices(ALL_CHARS[:36], k=random.randint(2, 5)))
        elif roll < 0.70:
            text = random.choice(WORDS)
        else:
            text = " ".join(random.choices(WORDS, k=random.randint(2, 3)))

        fw = random.choice([0, 18, 20]) if wpm < 18 else 0
        samples.append(dict(
            text=text,
            wpm=wpm,
            freq=random.uniform(400, 1100),
            noise=random.uniform(0.0, 0.25),
            fw=fw,
        ))

    random.shuffle(samples)
    metadata: list[dict] = []

    print(f"Generating {len(samples)} samples …")
    for i, p in enumerate(samples):
        audio = text_to_audio(p["text"], p["wpm"], p["freq"], p["noise"], p["fw"])
        fname = f"sample_{i:06d}.wav"
        wavfile.write(os.path.join(audio_dir, fname), SAMPLE_RATE, audio)
        metadata.append({
            "file":        fname,
            "text":        p["text"],
            "wpm":         p["wpm"],
            "freq":        round(p["freq"], 1),
            "noise":       round(p["noise"], 3),
            "fw_char_wpm": p["fw"],
        })
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(samples)}")

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    n_guaranteed = len(ALL_CHARS) * 12
    print(f"\nDataset saved to {output_dir}/")
    print(f"  Guaranteed coverage : {n_guaranteed}  ({len(ALL_CHARS)} chars × 12 conditions)")
    print(f"  Random samples      : {n_random}")
    print(f"  Total               : {len(samples)}")
    print("\nNext step: python train.py")


if __name__ == "__main__":
    generate_dataset()
