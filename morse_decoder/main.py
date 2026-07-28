"""
main.py — Entry Point
======================
Launches the live Morse Code Decoder dashboard.

Usage
-----
    python main.py                  # open GUI with no file pre-loaded
    python main.py myfile.wav       # open GUI and start decoding a WAV
    python main.py --mic            # open GUI and start microphone input
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    args      = sys.argv[1:]
    start_mic = "--mic" in args
    pos_args  = [a for a in args if not a.startswith("--")]
    wav_file  = pos_args[0] if pos_args else None

    if wav_file and not os.path.exists(wav_file):
        print(f"Error: file not found — {wav_file}")
        sys.exit(1)

    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("tkinter is not available on this system.")
        print("On Linux, run:  sudo apt-get install python3-tk")
        sys.exit(1)

    from src.ui import DecoderUI
    app = DecoderUI(initial_file=wav_file)
    if start_mic:
        app._root.after(400, app._start_mic)
    app.run()


if __name__ == "__main__":
    main()
