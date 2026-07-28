"""
debug_morse.py — WAV Generator + DSP Debug Script
==================================================
Generates a noisy Morse WAV file and prints a step-by-step trace of
how engine.py decodes it — useful for tuning thresholds and debugging.

Run:
    python debug_morse.py
Output:
    noisy_paragraph.wav  (generated)
    Full DSP trace to stdout
"""

from __future__ import annotations

import numpy as np
import scipy.io.wavfile as wav

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 8_000
SPEED_WPM   = 15
FREQUENCY   = 700
NOISE_AMP   = 2_000   # amplitude of additive Gaussian noise (out of 16384)
OUTPUT_WAV  = "noisy_paragraph.wav"

TEXT = (
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. "
    "MORSE CODE NINJA TRAINING IS ESSENTIAL FOR "
    "DEVELOPING HIGH SPEED TELEGRAPHY SKILLS."
)

MORSE_DICT: dict[str, str] = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",  "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    "0": "-----", " ": " ",    ".": ".-.-.-", ",": "--..--",
    "?": "..--..", "/": "-..-.","(": "-.--.", ")": "-.--.-", "-": "-....-",
}

MORSE_REV: dict[str, str] = {v: k for k, v in MORSE_DICT.items() if k != " "}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate noisy WAV
# ─────────────────────────────────────────────────────────────────────────────

def generate_wav() -> None:
    text = "".join(c for c in TEXT.upper() if c in MORSE_DICT)

    dot_dur     = 1.2 / SPEED_WPM
    dash_dur    = dot_dur * 3
    intra_dur   = dot_dur
    letter_dur  = dot_dur * 3
    word_dur    = dot_dur * 7

    audio: list[float] = []

    def tone(d: float) -> None:
        t = np.linspace(0, d, int(SAMPLE_RATE * d), endpoint=False)
        audio.extend((np.sin(2 * np.pi * FREQUENCY * t) * 16384).tolist())

    def silence(d: float) -> None:
        audio.extend([0.0] * int(SAMPLE_RATE * d))

    for i, ch in enumerate(text):
        if ch == " ":
            silence(word_dur)
            continue
        for j, sym in enumerate(MORSE_DICT[ch]):
            tone(dot_dur if sym == "." else dash_dur)
            if j < len(MORSE_DICT[ch]) - 1:
                silence(intra_dur)
        if i < len(text) - 1 and text[i + 1] != " ":
            silence(letter_dur)

    sig   = np.array(audio)
    noise = np.random.normal(0, NOISE_AMP, sig.shape)
    noisy = (sig + noise).astype(np.int16)
    wav.write(OUTPUT_WAV, SAMPLE_RATE, noisy)
    print(f"WAV written: {OUTPUT_WAV}  ({len(noisy) / SAMPLE_RATE:.1f} s)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DSP debug trace
# ─────────────────────────────────────────────────────────────────────────────

def debug_decode() -> None:
    sr, data = wav.read(OUTPUT_WAV)
    print(f"Read: sr={sr}  shape={data.shape}  dtype={data.dtype}")

    if data.ndim > 1:
        data = data[:, 0]

    amp       = np.abs(data)
    threshold = 0.15 * np.max(amp)
    print(f"Threshold: {threshold:.1f}   Max amplitude: {np.max(amp)}")

    binary = (amp > threshold).astype(int)
    print(f"Binary signal — ones: {binary.sum()}  length: {len(binary)}")

    # Run-length encode
    state   = binary[0]
    count   = 0
    tones   = []
    silences = []
    for bit in binary:
        if bit == state:
            count += 1
        else:
            dur = (count / sr) * 1000
            (tones if state == 1 else silences).append(dur)
            state, count = bit, 1
    if count:
        dur = (count / sr) * 1000
        (tones if state == 1 else silences).append(dur)

    print(f"Tone segments: {len(tones)}   Silence segments: {len(silences)}")
    print(f"Tones (first 20 ms):    {[f'{t:.1f}' for t in tones[:20]]}")
    print(f"Silences (first 20 ms): {[f'{s:.1f}' for s in silences[:20]]}")

    base_unit         = np.percentile(tones, 15)
    dot_dash_boundary = base_unit * 2
    letter_boundary   = base_unit * 5
    print(f"Base unit: {base_unit:.1f} ms   "
          f"Dit/Dah split: {dot_dash_boundary:.1f} ms   "
          f"Letter/Word split: {letter_boundary:.1f} ms")

    # Decode
    morse_str = ""
    for i in range(len(tones)):
        morse_str += "." if tones[i] < dot_dash_boundary else "-"
        if i < len(silences):
            if silences[i] >= letter_boundary:
                morse_str += "   "
            elif silences[i] >= dot_dash_boundary:
                morse_str += " "

    print(f"\nMorse string (repr): {morse_str!r}")

    words = morse_str.strip().split("   ")
    decoded = []
    for word in words:
        letters = [MORSE_REV.get(sym, "?") for sym in word.split(" ") if sym]
        decoded.append("".join(letters))

    print(f"\nDecoded words: {decoded}")
    print(f"Final text   : {' '.join(decoded)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Engine cross-check
# ─────────────────────────────────────────────────────────────────────────────

def engine_check() -> None:
    from src.engine import decode_wav
    result = decode_wav(OUTPUT_WAV)
    print(f"\nEngine.decode() → {result!r}")


if __name__ == "__main__":
    generate_wav()
    print()
    debug_decode()
    print()
    engine_check()
