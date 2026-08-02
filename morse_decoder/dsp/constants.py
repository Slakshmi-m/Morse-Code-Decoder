"""
constants.py — Standard Morse Code Mappings
============================================
Single source of truth for all Morse ↔ character mappings used
across engine.py, corrector.py, and generate_dataset.py.
"""

MORSE_MAP: dict[str, str] = {
    # Letters
    ".-":   "A", "-...": "B", "-.-.": "C", "-..":  "D", ".":    "E",
    "..-.": "F", "--.":  "G", "....": "H", "..":   "I", ".---": "J",
    "-.-":  "K", ".-..": "L", "--":   "M", "-.":   "N", "---":  "O",
    ".--.": "P", "--.-": "Q", ".-.":  "R", "...":  "S", "-":    "T",
    "..-":  "U", "...-": "V", ".--":  "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    # Digits
    ".----": "1", "..---": "2", "...--": "3", "....-": "4", ".....": "5",
    "-....": "6", "--...": "7", "---..": "8", "----.": "9", "-----": "0",
    # Punctuation
    ".-.-.-": ".", "--..--": ",", "---...": ":", "..--..": "?",
    ".----.": "'", "-....-": "-", "-..-.":  "/", ".-..-.": '"',
    ".--.-.": "@", "-...-":  "=", ".-.-.":  "+", "...-..-": "$",
    "-.-.--": "!",
    # Word space sentinel
    " ": " ",
}

# Inverse map (character → Morse) for encoding
TEXT_TO_MORSE: dict[str, str] = {v: k for k, v in MORSE_MAP.items()}

# Full Morse table used by dataset generator and inference
MORSE_TABLE: dict[str, str] = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",  "0": "-----", "1": ".----", "2": "..---", "3": "...--",
    "4": "....-", "5": ".....", "6": "-....", "7": "--...", "8": "---..",
    "9": "----.", ".": ".-.-.-", ",": "--..--", "?": "..--..", "-": "-....-",
    "/": "-..-.", "=": "-...-",
}

# All characters the ML model can decode (must match model.py VOCAB)
ALL_CHARS: list[str] = list(MORSE_TABLE.keys())
