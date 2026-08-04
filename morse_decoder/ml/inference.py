"""
inference.py — Decode a WAV File Using the Trained ML Model
============================================================
Usage:
    python inference.py <audio.wav>
    python inference.py <audio.wav> model_best.pt

Requires model_best.pt produced by train.py.
Falls back gracefully if the checkpoint is missing.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchaudio.transforms as T
from scipy.io import wavfile
from scipy.signal import butter, lfilter, welch

from ml.model import MorseDecoder, greedy_decode

_model_cache: dict = {}
_CHUNK_SECONDS = 8.0
_RECON_FREQ    = 700.0   # fixed carrier for reconstructed signal (centre of training range)


# ─────────────────────────────────────────────────────────────────────────────
# DSP preprocessing — cleans the signal before ML decoding
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(data: np.ndarray, sr: int, low: float, high: float,
              order: int = 5) -> np.ndarray:
    nyq  = 0.5 * sr
    b, a = butter(order, [max(0.01, low / nyq), min(0.99, high / nyq)], btype="band")
    return lfilter(b, a, data)


def _dsp_preprocess(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    DSP → clean binary reconstruction at _RECON_FREQ.

    Steps
    -----
    1. Detect dominant carrier via Welch PSD (300–1200 Hz).
    2. Bandpass ±200 Hz around carrier to strip wideband noise.
    3. Compute smoothed envelope; adaptive-threshold → binary ON/OFF.
    4. Filter noise spikes (segments shorter than rough_unit / 3).
    5. Synthesise a clean sine at _RECON_FREQ matching the ON/OFF timing.

    The model is trained on 700 Hz synthetic audio, so feeding it a clean
    700 Hz reconstruction (regardless of original carrier) keeps the mel
    spectrogram in-distribution.
    """
    float_data = samples.astype(np.float32) / 32768.0

    # 1. Carrier detection (400 Hz floor matches training data range)
    f, psd = welch(float_data, sr, nperseg=1024)
    mask   = (f >= 400) & (f < 1200)
    carrier = float(f[mask][np.argmax(psd[mask])]) if np.any(mask) else 700.0

    # 2. Bandpass
    filtered = _bandpass(float_data, sr, carrier - 200, carrier + 200)

    # 3. Envelope + normalise
    env  = np.abs(filtered)
    peak = np.max(env)
    if peak > 0:
        env = env / peak

    # 4. Smooth (10 ms box — longer window suppresses background music spikes)
    win    = max(1, int(sr * 0.010))
    smooth = np.convolve(env, np.ones(win) / win, mode="same")

    floor = np.percentile(smooth, 5)
    top   = np.percentile(smooth, 95)
    threshold = floor + (top - floor) * 0.45
    binary = (smooth > threshold).astype(np.int8)

    # 5. Run-length encode
    changes = np.diff(binary)
    idx     = np.concatenate(([0], np.where(changes != 0)[0] + 1, [len(binary)]))
    segs    = [(int(binary[idx[i]]), int(np.diff(idx)[i])) for i in range(len(idx) - 1)]

    # 6. Noise-spike filter
    # Absolute floor: 10 ms at 8 kHz = 80 samples — nothing shorter is real Morse
    # even at 120 WPM (competition speed). This kills music/background spikes.
    on_durs    = [d for s, d in segs if s == 1]
    if not on_durs:
        return samples          # can't clean — return as-is
    rough_unit = np.percentile(on_durs, 25)
    min_dur    = max(int(sr * 0.010), int(rough_unit / 3))
    segs       = [(s, d) for s, d in segs if d >= min_dur]

    # 7. Reconstruct clean 700 Hz audio
    out = np.zeros(len(samples), dtype=np.float32)
    pos = 0
    t0  = 0.0
    for state, dur in segs:
        end = min(pos + dur, len(out))
        if state == 1:
            t   = np.arange(dur, dtype=np.float32) / sr + t0
            out[pos:end] = np.sin(2 * np.pi * _RECON_FREQ * t[:end - pos])
            t0 += dur / sr
        else:
            t0 = 0.0        # reset phase on silence so next tone starts clean
        pos = end
        if pos >= len(out):
            break

    return (out * 16384).clip(-32767, 32767).astype(np.int16)


def _split_at_word_gaps(samples: np.ndarray, sr: int,
                        pad: int = 800) -> list[np.ndarray]:
    """
    Use DSP timing to split audio at inter-word gaps.

    Returns a list of sample arrays — one per word.  Falls back to the
    full buffer if no reliable word boundary is found.
    """
    float_data = samples.astype(np.float32) / 32768.0

    f, psd  = welch(float_data, sr, nperseg=1024)
    mask    = (f >= 400) & (f < 1200)
    carrier = float(f[mask][np.argmax(psd[mask])]) if np.any(mask) else 700.0

    filtered = _bandpass(float_data, sr, carrier - 200, carrier + 200)
    env      = np.abs(filtered)
    peak     = np.max(env)
    if peak == 0:
        return [samples]
    env = env / peak

    win    = max(1, int(sr * 0.010))
    smooth = np.convolve(env, np.ones(win) / win, mode="same")
    floor  = np.percentile(smooth, 5)
    top    = np.percentile(smooth, 95)
    binary = (smooth > floor + (top - floor) * 0.45).astype(np.int8)

    changes = np.diff(binary)
    idx     = np.concatenate(([0], np.where(changes != 0)[0] + 1, [len(binary)]))
    segs    = [(int(binary[idx[i]]), int(np.diff(idx)[i])) for i in range(len(idx) - 1)]

    on_durs = [d for s, d in segs if s == 1]
    if not on_durs:
        return [samples]

    rough_unit = float(np.percentile(on_durs, 25))
    word_thresh = rough_unit * 5.0

    # Collect split points at the centre of each word-gap OFF segment
    splits: list[int] = []
    pos = 0
    for state, dur in segs:
        if state == 0 and dur >= word_thresh:
            splits.append(pos + dur // 2)
        pos += dur

    if not splits:
        return [samples]

    chunks: list[np.ndarray] = []
    prev = 0
    for sp in splits:
        lo = max(prev, sp - pad)
        hi = min(len(samples), sp + pad)
        chunks.append(samples[prev:hi])
        prev = lo
    chunks.append(samples[prev:])
    return [c for c in chunks if len(c) >= sr // 8]


def _load_model(model_path: str, device: torch.device):
    key = (model_path, str(device))
    if key not in _model_cache:
        ckpt         = torch.load(model_path, map_location=device, weights_only=False)
        n_mels       = ckpt.get("n_mels", 32)
        cnn_channels = ckpt.get("cnn_channels", 64)
        hidden       = ckpt.get("hidden", 128)
        layers       = ckpt.get("layers", 2)
        model = MorseDecoder(n_mels=n_mels, cnn_channels=cnn_channels,
                             hidden=hidden, layers=layers).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache[key] = (model, n_mels)
    return _model_cache[key]


def _trim_silence(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Strip leading/trailing silence from a reconstructed int16 buffer.
    Prevents the model from seeing word-gap padding as signal,
    which it can misread as spurious characters (e.g. '5' = ·····).
    """
    env  = np.abs(samples.astype(np.float32))
    peak = np.max(env)
    if peak == 0:
        return samples
    smooth = np.convolve(env / peak,
                         np.ones(max(1, int(sr * 0.01))) / max(1, int(sr * 0.01)),
                         mode="same")
    above = np.where(smooth > 0.05)[0]
    if len(above) == 0:
        return samples
    margin = int(sr * 0.01)   # 10 ms keep-alive margin on each side
    return samples[max(0, above[0] - margin) : min(len(samples), above[-1] + margin)]


def _run_model(samples: np.ndarray, sr: int, model, n_mels: int, device) -> str:
    clean   = _dsp_preprocess(samples, sr)
    trimmed = _trim_silence(clean, sr)
    if len(trimmed) < sr // 20:   # < 50 ms after trim — nothing meaningful
        return ""
    audio = torch.FloatTensor(trimmed.astype(np.float32) / 32768.0).unsqueeze(0)
    mel   = T.MelSpectrogram(sample_rate=sr, n_fft=256, hop_length=64, n_mels=n_mels)
    spec  = T.AmplitudeToDB()(mel(audio)).unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(spec)[0]
    return greedy_decode(log_probs)


def _find_split(samples: np.ndarray, sr: int, target: int, search_radius: int) -> int:
    win  = max(1, sr // 20)
    step = max(1, win // 4)
    lo   = max(0, target - search_radius)
    hi   = min(len(samples) - win, target + search_radius)
    best_pos, best_energy = target, float("inf")
    for pos in range(lo, hi, step):
        e = float(np.sum(samples[pos: pos + win].astype(np.float32) ** 2))
        if e < best_energy:
            best_energy, best_pos = e, pos + win // 2
    return int(best_pos)


def _split_into_chunks(samples: np.ndarray, sr: int, max_chunk: int) -> list:
    if len(samples) <= max_chunk:
        return [samples]
    target        = max_chunk
    search_radius = max_chunk // 3    # wider search → more likely to land on silence
    split         = _find_split(samples, sr, target, search_radius)
    split         = max(sr // 2, min(split, len(samples) - sr // 2))
    return (_split_into_chunks(samples[:split], sr, max_chunk) +
            _split_into_chunks(samples[split:], sr, max_chunk))


def decode_buffer_ml(samples: np.ndarray, sr: int = 8_000,
                     model_path: str = "model_best.pt") -> str:
    """Decode a numpy int16 audio buffer using the trained CRNN model.

    Pipeline: DSP bandpass + binary reconstruction → mel spectrogram → CRNN.
    Word gaps detected by DSP timing split the buffer so each word is decoded
    independently, which matches how the model was trained (short samples).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, n_mels = _load_model(model_path, device)
    except FileNotFoundError:
        return f"[Checkpoint not found: {model_path} — run: python ml/train.py]"
    except Exception as exc:
        return f"[Model load error: {exc}]"

    max_chunk = int(_CHUNK_SECONDS * sr)

    # Split at word gaps first (DSP-guided), then apply the hard size cap
    word_chunks = _split_at_word_gaps(samples, sr)
    all_chunks: list[np.ndarray] = []
    for wc in word_chunks:
        if len(wc) <= max_chunk:
            all_chunks.append(wc)
        else:
            all_chunks.extend(_split_into_chunks(wc, sr, max_chunk))

    parts = [_run_model(c, sr, model, n_mels, device)
             for c in all_chunks if len(c) >= sr // 8]
    return " ".join(p for p in parts if p)


def decode_wav_ml(file_path: str, model_path: str = "model_best.pt",
                  sr: int = 8_000) -> str:
    """Decode a WAV file using the trained CRNN model.

    Handles any sample rate, stereo, int16/int32/float32 WAV files.
    """
    try:
        file_sr, raw = wavfile.read(file_path)

        # Stereo → mono
        if raw.ndim > 1:
            raw = raw[:, 0]

        # Normalise to float32 [-1, 1]
        if raw.dtype == np.int16:
            audio = raw.astype(np.float32) / 32768.0
        elif raw.dtype == np.int32:
            audio = raw.astype(np.float32) / 2_147_483_648.0
        else:
            audio = raw.astype(np.float32)

        # Resample to 8 000 Hz if the file uses a different rate
        if file_sr != sr:
            n     = int(len(audio) * sr / file_sr)
            audio = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(audio)),
                audio,
            ).astype(np.float32)

        # Back to int16 — decode_buffer_ml expects int16 samples
        samples = (audio * 32767).clip(-32767, 32767).astype(np.int16)
        return decode_buffer_ml(samples, sr=sr, model_path=model_path)

    except Exception as exc:
        return f"[File error: {exc}]"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio.wav> [model_best.pt]")
        sys.exit(1)
    wav = sys.argv[1]
    mdl = sys.argv[2] if len(sys.argv) > 2 else "model_best.pt"
    print(f"Decoded: {decode_wav_ml(wav, mdl)}")
