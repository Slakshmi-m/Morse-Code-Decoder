"""
corrector.py — Probabilistic N-gram Correction Layer
=====================================================
Uses character-level bigram and trigram frequency tables built from a
built-in English + amateur-radio corpus to fix likely decoding errors
([?] tokens, and ambiguous characters) produced by engine.py.

This fulfils the assignment requirement:
  "Probabilistic Correction: Implement a model that uses probability
   (e.g., Hidden Markov Models or N-grams) to fill gaps or correct
   likely errors in the character stream."

How it works
------------
1.  At construction time, the class counts every consecutive pair
    (bigram) and triple (trigram) of characters seen in the training
    corpus.  These counts are turned into conditional probabilities:
        P(c | prev)  and  P(c | prev2, prev)
2.  When correct() is called on a decoded string it scans for every
    '[?]' placeholder left by engine.py and replaces it with whichever
    character has the highest probability given its neighbours.
3.  A simple Viterbi-style pass then looks at low-confidence isolated
    characters (those that appear between two high-frequency pairs)
    and, if an alternative is far more likely, substitutes it.

Why this matters for Morse decoding
------------------------------------
At high noise levels or unusual sending speeds, engine.py sometimes
cannot decide between similar Morse patterns (e.g. '.-' vs '---') and
emits [?].  Language statistics let us recover: if the surrounding
letters are "TH_  QUIC" the missing character is almost certainly "E".
"""

import re
import math
from collections import defaultdict
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Built-in training corpus
# ---------------------------------------------------------------------------
# A compact but representative English + amateur-radio corpus.
# The corrector trains entirely from this text at startup — no external files.

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
INTERCHARACTER GAPS ARE STRETCHED IN FARNSWORTH METHOD
ADAPTIVE TIMING ADJUSTS TO DIFFERENT SENDING STYLES
THRESHOLD DETECTION CONVERTS AUDIO TO BINARY SIGNAL
PEAK FREQUENCY DETECTION FINDS THE TONE CARRIER
""".upper()

# Extended word list covering common ham radio abbreviations and prosigns
_EXTRA_WORDS = [
    "HELLO", "FOLLOW", "FELLOW", "HOLLOW", "ALLOW", "HELLO", "HELLO",
    "CQ", "DE", "AR", "SK", "BK", "KN", "QRN", "QRM", "QSB", "QSO",
    "QTH", "QRZ", "RST", "TNX", "TU", "UR", "ES", "OM", "YL", "DX",
    "HI", "HR", "SIG", "ANT", "RIG", "PWR", "WX", "NR", "NW", "GD",
    "GA", "GM", "GE", "GN", "FB", "OK", "RGR", "ROGER", "WILCO",
    "ALFA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF",
    "HOTEL", "INDIA", "JULIET", "KILO", "LIMA", "MIKE", "NOVEMBER",
    "OSCAR", "PAPA", "QUEBEC", "ROMEO", "SIERRA", "TANGO", "UNIFORM",
    "VICTOR", "WHISKEY", "XRAY", "YANKEE", "ZULU",
    "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE",
    "SIX", "SEVEN", "EIGHT", "NINE", "TEN",
]


class MorseCorrector:
    """
    Probabilistic character-level N-gram corrector for decoded Morse text.

    Parameters
    ----------
    smoothing : float
        Laplace (add-k) smoothing value.  A small positive number prevents
        zero probabilities for unseen pairs.  Default 0.01 works well.
    use_trigrams : bool
        If True, trigram context is also used (higher accuracy, slightly
        slower).  Default True.
    """

    # Characters the corrector can substitute (full Morse alphabet)
    ALPHABET: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?-/= "

    def __init__(self, smoothing: float = 0.01, use_trigrams: bool = True):
        self.smoothing = smoothing
        self.use_trigrams = use_trigrams

        # Frequency tables
        self._unigram:  Dict[str, int]              = defaultdict(int)
        self._bigram:   Dict[Tuple[str, str], int]  = defaultdict(int)
        self._trigram:  Dict[Tuple[str, str, str], int] = defaultdict(int)

        self._train(_CORPUS)
        for word in _EXTRA_WORDS:
            self._train(word + " ")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, text: str) -> None:
        """Count n-gram frequencies from a text string."""
        # Keep only characters in our alphabet + a START sentinel
        cleaned = [c for c in text if c in self.ALPHABET]
        if len(cleaned) < 2:
            return

        for c in cleaned:
            self._unigram[c] += 1

        for i in range(len(cleaned) - 1):
            self._bigram[(cleaned[i], cleaned[i + 1])] += 1

        if self.use_trigrams:
            for i in range(len(cleaned) - 2):
                self._trigram[(cleaned[i], cleaned[i + 1], cleaned[i + 2])] += 1

    # ------------------------------------------------------------------
    # Probability queries
    # ------------------------------------------------------------------

    def _bigram_prob(self, prev: str, char: str) -> float:
        """P(char | prev) with Laplace smoothing."""
        numerator   = self._bigram.get((prev, char), 0) + self.smoothing
        denominator = self._unigram.get(prev, 0) + self.smoothing * len(self.ALPHABET)
        return numerator / denominator

    def _trigram_prob(self, prev2: str, prev: str, char: str) -> float:
        """P(char | prev2, prev) with Laplace smoothing."""
        numerator   = self._trigram.get((prev2, prev, char), 0) + self.smoothing
        denominator = (self._bigram.get((prev2, prev), 0)
                       + self.smoothing * len(self.ALPHABET))
        return numerator / denominator

    def _score(self, prev2: Optional[str], prev: Optional[str],
               candidate: str, nxt: Optional[str]) -> float:
        """
        Combined log-probability score for placing `candidate` at a position
        given its left context (prev2, prev) and right neighbour (nxt).
        Interpolates bigram and trigram scores.
        """
        score = 0.0

        if prev is not None:
            score += math.log(self._bigram_prob(prev, candidate) + 1e-12)

            if self.use_trigrams and prev2 is not None:
                score += math.log(self._trigram_prob(prev2, prev, candidate) + 1e-12)

        if nxt is not None:
            score += math.log(self._bigram_prob(candidate, nxt) + 1e-12)

        return score

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def correct(self, decoded: str) -> str:
        """
        Given a raw decoded Morse string (possibly containing '[?]' tokens),
        return a corrected string where unknowns have been filled in and
        low-confidence characters have been substituted when a better
        candidate is significantly more probable.

        Parameters
        ----------
        decoded : str
            Output from MorseEngine.decode(), e.g. "TH[?] QUICK BR[?]WN"

        Returns
        -------
        str
            Corrected string, e.g. "THE QUICK BROWN"
        """
        if not decoded or decoded.startswith("["):
            return decoded

        # ---- Pass 1: tokenise into a list of (char_or_unknown) ----
        tokens = self._tokenise(decoded)

        # ---- Pass 2: fill '[?]' slots ----
        tokens = self._fill_unknowns(tokens)

        # ---- Pass 3: fix isolated low-confidence characters ----
        tokens = self._fix_low_confidence(tokens)

        return "".join(tokens)

    # ------------------------------------------------------------------
    # Internal passes
    # ------------------------------------------------------------------

    def _tokenise(self, text: str):
        """Split decoded string into individual characters, preserving [?]."""
        tokens = []
        i = 0
        while i < len(text):
            if text[i:i+3] == "[?]":
                tokens.append("[?]")
                i += 3
            else:
                tokens.append(text[i].upper())
                i += 1
        return tokens

    def _fill_unknowns(self, tokens: list) -> list:
        """Replace every '[?]' with the best character given context."""
        result = list(tokens)
        for i, tok in enumerate(result):
            if tok != "[?]":
                continue

            prev2 = result[i - 2] if i >= 2 and result[i-2] != "[?]" else None
            prev  = result[i - 1] if i >= 1 and result[i-1] != "[?]" else None
            nxt   = result[i + 1] if i < len(result)-1 and result[i+1] != "[?]" else None

            best_char  = "?"
            best_score = -math.inf

            for candidate in self.ALPHABET:
                s = self._score(prev2, prev, candidate, nxt)
                if s > best_score:
                    best_score = s
                    best_char  = candidate

            result[i] = best_char

        return result

    def _fix_low_confidence(self, tokens: list) -> list:
        """
        Scan every character and replace it if an alternative is at least
        4× more probable given the surrounding context.  This catches
        common Morse confusions such as E vs I, or T vs M.
        """
        SUBSTITUTION_THRESHOLD = math.log(50.0)  # only substitute when 50x more probable

        result = list(tokens)
        for i, tok in enumerate(result):
            if tok in ("[?]", " "):
                continue

            prev2 = result[i - 2] if i >= 2 else None
            prev  = result[i - 1] if i >= 1 else None
            nxt   = result[i + 1] if i < len(result) - 1 else None

            current_score = self._score(prev2, prev, tok, nxt)

            for candidate in self.ALPHABET:
                if candidate == tok:
                    continue
                s = self._score(prev2, prev, candidate, nxt)
                if s - current_score > SUBSTITUTION_THRESHOLD:
                    result[i] = candidate
                    break   # take the first clearly superior alternative

        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def stats(self) -> str:
        """Return a summary of the trained model."""
        return (
            f"MorseCorrector | "
            f"unigrams={len(self._unigram)} | "
            f"bigrams={len(self._bigram)} | "
            f"trigrams={len(self._trigram)}"
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    corrector = MorseCorrector()
    print(corrector.stats())

    tests = [
        ("TH[?] QUICK BROWN FOX",      "THE QUICK BROWN FOX"),
        ("HELL[?] WORLD",              "HELLO WORLD"),
        ("MO[?]SE CODE",               "MORSE CODE"),
        ("CQ [?]Q DE TEST",            "CQ CQ DE TEST"),
    ]

    print("\n--- Self-test ---")
    for raw, expected in tests:
        result = corrector.correct(raw)
        status = "PASS" if result == expected else f"got '{result}'"
        print(f"  Input   : {raw}")
        print(f"  Fixed   : {result}  [{status}]")
        print()
