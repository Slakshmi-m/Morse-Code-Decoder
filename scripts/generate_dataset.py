"""
Generates a labeled Morse code training dataset.
Covers all 42 characters at multiple speeds, frequencies, and noise levels,
including Farnsworth spacing (as used by Morse Code Ninja).

Run: python generate_dataset.py
Output: dataset/audio/*.wav + dataset/metadata.json
"""

import json
import os
import random

import numpy as np
from scipy.io import wavfile

# All characters the model will learn — must stay in sync with model.py VOCAB
MORSE_TABLE = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',   'E': '.',
    'F': '..-.',  'G': '--.',   'H': '....',  'I': '..',    'J': '.---',
    'K': '-.-',   'L': '.-..',  'M': '--',    'N': '-.',    'O': '---',
    'P': '.--.',  'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',  'Y': '-.--',
    'Z': '--..',  '0': '-----', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.',  '.': '.-.-.-', ',': '--..--', '?': '..--..', '-': '-....-',
    '/': '-..-.',  '=': '-...-',
}

SAMPLE_RATE = 8000
ALL_CHARS = list(MORSE_TABLE.keys())

# Common words and amateur radio Q-codes for realistic multi-char training
WORDS = [
    'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
    'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW',
    'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID',
    'SOS', 'CQ', 'DE', 'PARIS', 'MORSE', 'CODE', 'NINJA', 'TEST', 'HELLO',
    'WORLD', '73', '88', 'CW', 'TNX', 'AR', 'SK', 'QRN', 'QRM', 'HAM',
    'RADIO', 'CALL', 'SIGN', 'FREQ', 'BAND', 'METER', 'OM', 'YL', 'ES',
]


def text_to_audio(text: str, wpm: int, freq: float, noise: float,
                  fw_char_wpm: int = 0) -> np.ndarray:
    """
    Convert text to a Morse code audio waveform.

    wpm            : overall sending speed (words per minute)
    freq           : tone frequency in Hz
    noise          : noise amplitude as fraction of signal (0 = clean)
    fw_char_wpm    : Farnsworth character speed — characters are sent faster
                     than the overall wpm to create stretched letter/word gaps,
                     matching the style used by Morse Code Ninja.
    """
    char_wpm = fw_char_wpm if fw_char_wpm > wpm else wpm
    dot = 1.2 / char_wpm   # dot duration in seconds at character speed
    intra = dot             # gap between symbols within a character

    if char_wpm > wpm:
        # Farnsworth: stretch letter and word gaps to hit the slower overall wpm.
        # "PARIS" = 50 units; total time at char speed vs overall speed gives extra time.
        extra_per_unit = max(0.0, (60.0 / wpm - 50 * dot) / 19)
        letter_gap = dot * 3 + extra_per_unit * 3
        word_gap   = dot * 7 + extra_per_unit * 7
    else:
        letter_gap = dot * 3
        word_gap   = dot * 7

    audio: list = []

    def tone(d: float):
        n = int(SAMPLE_RATE * d)
        t = np.linspace(0, d, n, endpoint=False)
        audio.extend(np.sin(2 * np.pi * freq * t) * 16384)

    def silence(d: float):
        audio.extend(np.zeros(int(SAMPLE_RATE * d)))

    words = text.upper().split(' ')
    for wi, word in enumerate(words):
        chars = [c for c in word if c in MORSE_TABLE]
        for ci, ch in enumerate(chars):
            syms = MORSE_TABLE[ch]
            for si, s in enumerate(syms):
                tone(dot if s == '.' else dot * 3)
                if si < len(syms) - 1:
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


def generate_dataset(output_dir: str = 'data/synthetic', n_random: int = 500):
    audio_dir = os.path.join(output_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)

    samples = []

    # --- Guaranteed coverage ---
    # Every character × 6 WPM settings × 2 frequencies = 12 samples per char
    for char in ALL_CHARS:
        for wpm in [5, 10, 15, 20, 25, 35]:
            for freq in [600, 800]:
                samples.append(dict(
                    text=char, wpm=wpm, freq=float(freq),
                    noise=random.uniform(0.0, 0.12), fw=0
                ))

    # --- Random samples (varied content, speed, noise, Farnsworth) ---
    for _ in range(n_random):
        wpm = random.randint(5, 35)
        roll = random.random()
        if roll < 0.20:
            text = random.choice(ALL_CHARS)
        elif roll < 0.40:
            # Random character group (like Morse Code Ninja exercises)
            text = ''.join(random.choices(ALL_CHARS[:36], k=random.randint(2, 5)))
        elif roll < 0.70:
            text = random.choice(WORDS)
        else:
            text = ' '.join(random.choices(WORDS, k=random.randint(2, 3)))

        # Farnsworth spacing only makes sense when overall WPM < char WPM
        fw = random.choice([0, 18, 20]) if wpm < 18 else 0
        samples.append(dict(
            text=text,
            wpm=wpm,
            freq=random.uniform(400, 1100),
            noise=random.uniform(0.0, 0.25),
            fw=fw,
        ))

    random.shuffle(samples)
    metadata = []

    print(f'Generating {len(samples)} samples...')
    for i, p in enumerate(samples):
        audio = text_to_audio(p['text'], p['wpm'], p['freq'], p['noise'], p['fw'])
        fname = f'sample_{i:06d}.wav'
        wavfile.write(os.path.join(audio_dir, fname), SAMPLE_RATE, audio)
        metadata.append({
            'file': fname,
            'text': p['text'],
            'wpm': p['wpm'],
            'freq': round(p['freq'], 1),
            'noise': round(p['noise'], 3),
            'fw_char_wpm': p['fw'],
        })
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(samples)}')

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    n_guaranteed = len(ALL_CHARS) * 12
    print(f'\nDataset saved to {output_dir}/')
    print(f'  Guaranteed coverage : {n_guaranteed} ({len(ALL_CHARS)} chars × 12 conditions)')
    print(f'  Random samples      : {n_random}')
    print(f'  Total               : {len(samples)}')


if __name__ == '__main__':
    generate_dataset()
