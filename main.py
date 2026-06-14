"""
main.py — Entry Point
======================
Launches the all-in-one live dashboard.

Usage
-----
    python main.py                  # open GUI, no file pre-loaded
    python main.py myfile.wav       # open GUI and start decoding immediately
    python main.py --mic            # open GUI and start microphone immediately
"""

import os, sys


def main():
    args      = sys.argv[1:]
    start_mic = "--mic" in args
    pos       = [a for a in args if not a.startswith("--")]
    wav_file  = pos[0] if pos else None

    # Verify the file exists before launching the window
    if wav_file and not os.path.exists(wav_file):
        print(f"Error: file not found — {wav_file}")
        sys.exit(1)

    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("tkinter is not available on this system.")
        print("On Linux run:  sudo apt-get install python3-tk")
        sys.exit(1)

    from ui import DecoderUI
    app = DecoderUI(initial_file=wav_file)
    if start_mic:
        app._root.after(400, app._start_mic)
    app.run()


if __name__ == "__main__":
    main()
