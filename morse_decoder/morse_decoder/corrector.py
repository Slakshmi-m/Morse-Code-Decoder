"""
corrector.py — Probabilistic N-gram Correction Layer
=====================================================
Assignment requirement fulfilled:
  "Probabilistic Correction: Implement a model that uses probability
   (e.g., Hidden Markov Models or N-grams) to fill gaps or correct
   likely errors in the character stream."

How it works
------------
1. At construction time the class builds character-level bigram and
   trigram frequency tables from a built-in English + amateur-radio
   corpus (including standard prosigns CQ, DE, AR, SK, K and common
   Q-codes).  These are turned into conditional probabilities:
       P(c | prev)        — bigram
       P(c | prev2, prev) — trigram
2. correct() scans every "[?]" token left by engine.py and replaces it
   with the highest-probability character given its neighbours.
3. A second pass optionally substitutes confirmed characters when an
   alternative is at least 50× more probable — catching common Morse
   confusions like E↔I or T↔M.

Why N-grams for Morse
---------------------
Morse errors appear as entire character substitutions, not individual
bit flips.  Character-level N-grams trained on a representative corpus
are therefore the correct probabilistic tool: they capture which
letters naturally follow each other ("TH", "THE", "CQ DE") and can
fill gaps reliably even in weak-signal conditions.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Training corpus — English prose + amateur-radio patterns
# ─────────────────────────────────────────────────────────────────────────────

_CORPUS = """
THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
MORSE CODE IS A METHOD OF TRANSMITTING TEXT INFORMATION
AS A SERIES OF ON OFF TONES LIGHTS OR CLICKS
AMATEUR RADIO OPERATORS USE MORSE CODE ON THE AIR
CQ CQ CQ DE CALLSIGN CALLSIGN K
THE WEATHER IS FINE TODAY AND THE BANDS ARE OPEN
SIGNAL REPORT IS FIVE NINE IN THE LOG
QSO WITH MANY STATIONS AROUND THE WORLD TODAY
HELLO WORLD THIS IS A TEST OF THE MORSE DECODER
GOOD MORNING AND GOOD EVENING TO ALL STATIONS
FREQUENCY IS FOURTEEN MEGAHERTZ ON THE TWENTY METER BAND
ANTENNA IS A DIPOLE AT TEN METERS HIGH
POWER IS ONE HUNDRED WATTS INTO FIFTY OHM COAX
TRANSCEIVER IS WORKING WELL WITH LOW NOISE FLOOR
CONTEST EXCHANGE IS FIVE NINE AND SERIAL NUMBER
THANKS FOR THE CONTACT AND SEVENTY THREE
THE QUICK BROWN FOX JUMPED OVER THE LAZY SLEEPING DOG
PRACTICE MAKES PERFECT WHEN LEARNING MORSE CODE
INTERNATIONAL MORSE CODE USES DOTS AND DASHES
EACH LETTER IS SEPARATED BY A SHORT PAUSE
WORDS ARE SEPARATED BY A LONGER PAUSE BETWEEN THEM
STANDARD SPEED IS MEASURED IN WORDS PER MINUTE
BEGINNERS START AT FIVE WORDS PER MINUTE SPEED
EXPERTS CAN COPY OVER THIRTY WORDS PER MINUTE
THE PARIS WORD IS USED AS THE STANDARD MEASURE
SIGNAL TO NOISE RATIO AFFECTS DECODING ACCURACY
BANDPASS FILTER REMOVES UNWANTED INTERFERENCE NOISE
DIGITAL SIGNAL PROCESSING HELPS DECODE WEAK SIGNALS
MACHINE LEARNING CAN IMPROVE RECOGNITION ACCURACY
CONVOLUTIONAL NEURAL NETWORK EXTRACTS AUDIO FEATURES
LONG SHORT TERM MEMORY HANDLES SEQUENTIAL DATA WELL
CTC LOSS FUNCTION ALIGNS OUTPUT WITH INPUT SEQUENCE
PYTHON IS A GREAT LANGUAGE FOR SIGNAL PROCESSING WORK
NUMPY AND SCIPY PROVIDE FAST ARRAY OPERATIONS
TRAINING DATA IS GENERATED AT MULTIPLE SPEEDS
NOISE AUGMENTATION IMPROVES MODEL ROBUSTNESS GREATLY
FARNSWORTH SPACING USES FASTER CHARACTER SPEEDS
ADAPTIVE TIMING ADJUSTS TO DIFFERENT SENDING STYLES
THRESHOLD DETECTION CONVERTS AUDIO TO BINARY SIGNAL
PEAK FREQUENCY DETECTION FINDS THE TONE CARRIER
""".upper()

_EXTRA_WORDS: list[str] = [
    # Ham radio prosigns and abbreviations
    "CQ", "DE", "AR", "SK", "BK", "KN", "QRN", "QRM", "QSB", "QSO",
    "QTH", "QRZ", "RST", "TNX", "TU", "UR", "ES", "OM", "YL", "DX",
    "HI", "HR", "SIG", "ANT", "RIG", "PWR", "WX", "NR", "NW", "GD",
    "GA", "GM", "GE", "GN", "FB", "OK", "RGR", "ROGER", "WILCO",
    # NATO phonetic alphabet
    "ALFA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF",
    "HOTEL", "INDIA", "JULIET", "KILO", "LIMA", "MIKE", "NOVEMBER",
    "OSCAR", "PAPA", "QUEBEC", "ROMEO", "SIERRA", "TANGO", "UNIFORM",
    "VICTOR", "WHISKEY", "XRAY", "YANKEE", "ZULU",
    # Spelled-out numbers
    "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE",
    "SIX", "SEVEN", "EIGHT", "NINE", "TEN",
]

# Dataset metadata files the corrector will auto-load at startup.
# Every "text" label in these files becomes N-gram training data —
# no need to manually add words here when the dataset already has them.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET_META_PATHS: list[str] = [
    os.path.join(_HERE, "dataset",       "metadata.json"),
    os.path.join(_HERE, "ninja_dataset", "metadata.json"),
]


class MorseCorrector:
    """
    Probabilistic character-level N-gram corrector for decoded Morse text.

    Parameters
    ----------
    smoothing : float
        Laplace (add-k) smoothing.  Prevents zero probabilities for unseen
        pairs.  Default 0.01 is appropriate for this vocabulary size.
    use_trigrams : bool
        Whether to include trigram context (higher accuracy, marginally
        slower).  Default True.
    """

    # Full Morse decodable alphabet — space included as a valid substitute
    ALPHABET: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?-/= "

    def __init__(self, smoothing: float = 0.01, use_trigrams: bool = True) -> None:
        self.smoothing    = smoothing
        self.use_trigrams = use_trigrams

        self._unigram: Dict[str, int]                  = defaultdict(int)
        self._bigram:  Dict[Tuple[str, str], int]      = defaultdict(int)
        self._trigram: Dict[Tuple[str, str, str], int] = defaultdict(int)

        # Train on built-in corpus and static word list
        self._train(_CORPUS)
        for word in _EXTRA_WORDS:
            self._train(word + " ")

        # Auto-load every text label from available dataset metadata files.
        # This covers all words the Ninja/synthetic datasets contain without
        # needing to manually maintain a word list here.
        for meta_path in _DATASET_META_PATHS:
            self._train_from_metadata(meta_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def _train_from_metadata(self, meta_path: str) -> None:
        """Load every text label from a dataset metadata.json and train on it."""
        if not os.path.exists(meta_path):
            return
        try:
            with open(meta_path, encoding="utf-8") as f:
                records = json.load(f)
            for record in records:
                text = record.get("text", "")
                if text:
                    self._train(text + " ")
            print(f"[MorseCorrector] Loaded {len(records)} labels from {meta_path}")
        except Exception as exc:
            print(f"[MorseCorrector] Could not load {meta_path}: {exc}")

    def _train(self, text: str) -> None:
        """Count n-gram frequencies from a text string."""
        cleaned = [c for c in text.upper() if c in self.ALPHABET]
        if len(cleaned) < 2:
            return

        for c in cleaned:
            self._unigram[c] += 1

        for i in range(len(cleaned) - 1):
            self._bigram[(cleaned[i], cleaned[i + 1])] += 1

        if self.use_trigrams:
            for i in range(len(cleaned) - 2):
                self._trigram[(cleaned[i], cleaned[i + 1], cleaned[i + 2])] += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Probability queries
    # ──────────────────────────────────────────────────────────────────────────

    def _bigram_prob(self, prev: str, char: str) -> float:
        """P(char | prev) with Laplace smoothing."""
        num = self._bigram.get((prev, char), 0) + self.smoothing
        den = self._unigram.get(prev, 0) + self.smoothing * len(self.ALPHABET)
        return num / den

    def _trigram_prob(self, prev2: str, prev: str, char: str) -> float:
        """P(char | prev2, prev) with Laplace smoothing."""
        num = self._trigram.get((prev2, prev, char), 0) + self.smoothing
        den = (self._bigram.get((prev2, prev), 0)
               + self.smoothing * len(self.ALPHABET))
        return num / den

    def _score(self, prev2: Optional[str], prev: Optional[str],
               candidate: str, nxt: Optional[str]) -> float:
        """
        Combined log-probability for placing `candidate` at a position
        given left context (prev2, prev) and right neighbour (nxt).
        Interpolates bigram and trigram contributions.
        """
        score = 0.0
        if prev is not None:
            score += math.log(self._bigram_prob(prev, candidate) + 1e-12)
            if self.use_trigrams and prev2 is not None:
                score += math.log(
                    self._trigram_prob(prev2, prev, candidate) + 1e-12
                )
        if nxt is not None:
            score += math.log(self._bigram_prob(candidate, nxt) + 1e-12)
        return score

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def correct(self, decoded: str) -> str:
        """
        Fill "[?]" tokens and optionally fix low-confidence characters.

        Parameters
        ----------
        decoded : str
            Raw output from MorseEngine.decode(), e.g. "TH[?] QUICK BR[?]WN"

        Returns
        -------
        str
            Corrected string, e.g. "THE QUICK BROWN"
        """
        if not decoded or decoded.startswith("["):
            return decoded

        tokens = self._tokenise(decoded)
        tokens = self._fill_unknowns(tokens)
        tokens = self._fix_low_confidence(tokens)
        return "".join(tokens)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal passes
    # ──────────────────────────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> list[str]:
        """Split decoded string into individual chars, preserving [?]."""
        tokens: list[str] = []
        i = 0
        while i < len(text):
            if text[i:i + 3] == "[?]":
                tokens.append("[?]")
                i += 3
            else:
                tokens.append(text[i].upper())
                i += 1
        return tokens

    @staticmethod
    def _nearest_char(tokens: list[str], i: int, step: int) -> tuple[Optional[str], int]:
        """Walk in direction `step` from i, skipping spaces and [?]. Returns (char, pos)."""
        j = i + step
        while 0 <= j < len(tokens):
            if tokens[j] not in (" ", "[?]"):
                return tokens[j], j
            j += step
        return None, -1

    def _fill_unknowns(self, tokens: list[str]) -> list[str]:
        """Replace every '[?]' with the highest-probability character."""
        result = list(tokens)
        for i, tok in enumerate(result):
            if tok != "[?]":
                continue
            prev,  prev_pos  = self._nearest_char(result, i,        -1)
            prev2, _         = self._nearest_char(result, prev_pos,  -1) if prev_pos >= 0 else (None, -1)
            nxt,   _         = self._nearest_char(result, i,         +1)

            best_char, best_score = "?", -math.inf
            for candidate in self.ALPHABET:
                s = self._score(prev2, prev, candidate, nxt)
                if s > best_score:
                    best_score, best_char = s, candidate

            result[i] = best_char
        return result

    def _fix_low_confidence(self, tokens: list[str]) -> list[str]:
        """
        Substitute characters that are at least 20× less probable than
        an alternative — catches common Morse confusions (E↔I, T↔M, H↔N, etc.).
        Spaces are skipped when gathering N-gram context so that characters
        separated by word gaps still benefit from their neighbours.
        """
        THRESHOLD = math.log(20.0)
        result = list(tokens)
        for i, tok in enumerate(result):
            if tok in ("[?]", " "):
                continue
            prev,  prev_pos = self._nearest_char(result, i,       -1)
            prev2, _        = self._nearest_char(result, prev_pos, -1) if prev_pos >= 0 else (None, -1)
            nxt,   _        = self._nearest_char(result, i,        +1)

            current_score = self._score(prev2, prev, tok, nxt)
            for candidate in self.ALPHABET:
                if candidate == tok:
                    continue
                s = self._score(prev2, prev, candidate, nxt)
                if s - current_score > THRESHOLD:
                    result[i] = candidate
                    break
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> str:
        """Return a one-line summary of trained model sizes."""
        return (
            f"MorseCorrector | "
            f"unigrams={len(self._unigram)} | "
            f"bigrams={len(self._bigram)} | "
            f"trigrams={len(self._trigram)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    corrector = MorseCorrector()
    print(corrector.stats())

    tests = [
        ("TH[?] QUICK BROWN FOX", "THE QUICK BROWN FOX"),
        ("HELL[?] WORLD",         "HELLO WORLD"),
        ("MO[?]SE CODE",          "MORSE CODE"),
        ("CQ [?]Q DE TEST",       "CQ CQ DE TEST"),
    ]

    print("\n--- Self-test ---")
    for raw, expected in tests:
        result = corrector.correct(raw)
        status = "PASS" if result == expected else f"got '{result}'"
        print(f"  Input : {raw}")
        print(f"  Fixed : {result}  [{status}]\n")
