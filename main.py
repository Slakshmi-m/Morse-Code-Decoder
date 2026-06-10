import os
import sys

from engine import decode_wav


def main():
    print("=== Morse Code Decoder ===")

    audio_file = sys.argv[1] if len(sys.argv) > 1 else "noisy_paragraph.wav"

    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found.")
        print("Usage: python main.py <audio_file.wav>")
        return

    # Use ML model if a trained checkpoint exists, otherwise fall back to
    # the signal-processing engine (engine.py).
    if os.path.exists("model_best.pt"):
        try:
            from inference import decode_wav_ml
            print(f"ML model -> {audio_file}")
            result = decode_wav_ml(audio_file)
        except Exception as e:
            print(f"ML model failed ({e}), falling back to signal processing...")
            result = decode_wav(audio_file)
    else:
        print(f"Signal processing -> {audio_file}")
        print("Tip: run generate_dataset.py then train.py to enable the ML model.")
        result = decode_wav(audio_file)

    print(f"\nDecoded:\n{'-' * 20}\n{result}\n{'-' * 20}")


if __name__ == "__main__":
    main()