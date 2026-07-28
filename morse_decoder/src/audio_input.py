"""
audio_input.py — Audio Input (WAV File / Microphone / System Audio)
====================================================================
Three acquisition modes:
  1. WAV file streaming  — stream_file() / stream_file_async()
  2. Microphone          — start_microphone()  (via sounddevice)
  3. System audio        — start_system_audio() (via pyaudiowpatch WASAPI
                           loopback — captures speakers/headphones on Windows
                           without needing Stereo Mix)

Install once:
    pip install sounddevice pyaudiowpatch
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import numpy as np
from scipy.io import wavfile

# ── sounddevice (mic + WAV) ───────────────────────────────────────────────────
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

# ── pyaudiowpatch (WASAPI loopback — Windows system audio) ───────────────────
try:
    import pyaudiowpatch as pyaudio
    _PAW_AVAILABLE = True
except ImportError:
    _PAW_AVAILABLE = False

# Keyword list for loopback-device auto-detection via sounddevice
_LOOPBACK_KEYWORDS = [
    "stereo mix", "what u hear", "wave out mix",
    "cable output", "vb-audio", "virtual audio cable",
    "loopback", "mix output", "blackhole", "soundflower",
]

ChunkCallback = Callable[[np.ndarray, int], None]


def list_input_devices() -> list[tuple[int, str]]:
    """Return list of (index, name) for all available input devices."""
    if not _SD_AVAILABLE:
        return []
    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


class AudioInput:
    """
    Unified audio acquisition class.

    Registered callbacks receive (samples: np.ndarray[int16], sample_rate: int)
    for each chunk delivered by whichever input mode is active.
    """

    SAMPLE_RATE    = 8_000   # Hz — target rate for engine.py
    CHUNK_DURATION = 0.1     # seconds per delivered chunk

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 chunk_duration: float = CHUNK_DURATION,
                 device: Optional[int] = None) -> None:
        self.sample_rate    = sample_rate
        self.chunk_duration = chunk_duration
        self.device         = device
        self.chunk_size     = int(sample_rate * chunk_duration)

        self._callbacks: List[ChunkCallback] = []
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._on_error: Optional[Callable[[str], None]] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Callback management
    # ──────────────────────────────────────────────────────────────────────────

    def register_callback(self, fn: ChunkCallback) -> None:
        """Register a function called with (samples, sample_rate) per chunk."""
        self._callbacks.append(fn)

    def _dispatch(self, samples: np.ndarray) -> None:
        for cb in self._callbacks:
            try:
                cb(samples, self.sample_rate)
            except Exception as exc:
                print(f"[AudioInput] Callback error: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # Mode 1 — Microphone (sounddevice)
    # ──────────────────────────────────────────────────────────────────────────

    def start_microphone(self, device: Optional[int] = None) -> None:
        """Start capturing from the microphone."""
        if not _SD_AVAILABLE:
            raise RuntimeError("sounddevice not installed — run: pip install sounddevice")
        dev = device if device is not None else self.device
        self._running = True
        self._thread  = threading.Thread(
            target=self._sd_stream_loop, args=(dev,),
            daemon=True, name="mic",
        )
        self._thread.start()
        name = (sd.query_devices(dev)["name"] if dev is not None
                else sd.query_devices(kind="input")["name"])
        print(f"[AudioInput] Microphone: {name}")

    def _sd_stream_loop(self, device: Optional[int]) -> None:
        try:
            info      = (sd.query_devices(device) if device is not None
                         else sd.query_devices(kind="input"))
            native_sr = int(info["default_samplerate"])
            channels  = min(2, max(1, int(info["max_input_channels"])))
            blk       = int(native_sr * self.chunk_duration)

            with sd.InputStream(
                samplerate=native_sr, channels=channels,
                dtype="float32", device=device,
                blocksize=blk, latency="low",
            ) as stream:
                print("[AudioInput] Microphone stream open ✓")
                while self._running:
                    chunk, _ = stream.read(blk)
                    mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.ravel()
                    if native_sr != self.sample_rate:
                        mono = self._resample(mono, native_sr, self.sample_rate)
                    self._dispatch(
                        (mono * 32767).clip(-32767, 32767).astype(np.int16)
                    )
        except Exception as exc:
            print(f"[AudioInput] Microphone error: {exc}")
            self._running = False

    # ──────────────────────────────────────────────────────────────────────────
    # Mode 2 — System audio (WASAPI loopback via pyaudiowpatch)
    # ──────────────────────────────────────────────────────────────────────────

    def start_system_audio(self) -> str:
        """
        Capture everything playing through speakers/headphones.

        Uses WASAPI loopback on Windows — no Stereo Mix required.
        Raises RuntimeError with a sentinel string if pyaudiowpatch is absent.
        """
        if not _PAW_AVAILABLE:
            raise RuntimeError("NEED_PAWPATCH")
        self._running = True
        self._thread  = threading.Thread(
            target=self._wasapi_loop, daemon=True, name="wasapi-loopback"
        )
        self._thread.start()
        return "WASAPI Loopback"

    def _wasapi_loop(self) -> None:
        try:
            p = pyaudio.PyAudio()
            default_speakers = p.get_default_wasapi_loopback()
            if default_speakers is None:
                raise RuntimeError("No WASAPI loopback device found")

            native_sr = int(default_speakers["defaultSampleRate"])
            channels  = min(2, int(default_speakers["maxInputChannels"]))
            dev_idx   = default_speakers["index"]
            frames    = int(native_sr * self.chunk_duration)

            print(
                f"[AudioInput] WASAPI loopback: {default_speakers['name']!r} "
                f"sr={native_sr} ch={channels}"
            )

            stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=native_sr,
                input=True,
                input_device_index=dev_idx,
                frames_per_buffer=frames,
            )
            print("[AudioInput] WASAPI loopback stream open ✓")

            while self._running:
                raw = stream.read(frames, exception_on_overflow=False)
                arr = np.frombuffer(raw, dtype=np.float32).copy()
                if channels == 2:
                    arr = arr.reshape(-1, 2).mean(axis=1)
                if native_sr != self.sample_rate:
                    arr = self._resample(arr, native_sr, self.sample_rate)
                self._dispatch(
                    (arr * 32767).clip(-32767, 32767).astype(np.int16)
                )

            stream.stop_stream()
            stream.close()
            p.terminate()

        except Exception as exc:
            print(f"[AudioInput] WASAPI error: {exc}")
            self._running = False
            if callable(self._on_error):
                self._on_error(str(exc))

    # ──────────────────────────────────────────────────────────────────────────
    # Mode 3 — WAV file
    # ──────────────────────────────────────────────────────────────────────────

    def stream_file(self, path: str, realtime: bool = False) -> None:
        """Stream a WAV file chunk-by-chunk (blocking)."""
        try:
            fs, raw = wavfile.read(path)
        except Exception as exc:
            print(f"[AudioInput] Cannot read '{path}': {exc}")
            return

        if raw.ndim > 1:
            raw = raw[:, 0]
        if fs != self.sample_rate:
            raw = self._resample(
                raw.astype(np.float32), fs, self.sample_rate
            ).astype(np.int16)

        print(f"[AudioInput] Streaming '{path}' ({len(raw) / self.sample_rate:.1f} s)")
        self._running = True
        offset = 0
        while self._running and offset < len(raw):
            chunk = raw[offset: offset + self.chunk_size]
            if not len(chunk):
                break
            self._dispatch(chunk)
            offset += self.chunk_size
            if realtime:
                time.sleep(self.chunk_duration)
        self._running = False

    def stream_file_async(self, path: str, realtime: bool = True) -> None:
        """Non-blocking version of stream_file()."""
        self._thread = threading.Thread(
            target=self.stream_file, args=(path, realtime),
            daemon=True, name="file",
        )
        self._thread.start()

    # ──────────────────────────────────────────────────────────────────────────
    # Stop
    # ──────────────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal all capture threads to stop and wait for them."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resample(data: np.ndarray, orig: int, target: int) -> np.ndarray:
        """Linear interpolation resampler (lightweight, no extra dependencies)."""
        if orig == target:
            return data
        n = int(len(data) * target / orig)
        return np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(data)),
            data,
        ).astype(data.dtype)

    @staticmethod
    def list_devices() -> None:
        """Print all available input devices and WASAPI loopback status."""
        print("\n=== Input Devices ===")
        if _SD_AVAILABLE:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    tag = " ← LOOPBACK" if any(
                        kw in d["name"].lower() for kw in _LOOPBACK_KEYWORDS
                    ) else ""
                    print(f"  [{i:2d}] {d['name']}{tag}")
        else:
            print("  sounddevice not installed")

        print(f"\npyaudiowpatch available: {_PAW_AVAILABLE}")
        if _PAW_AVAILABLE:
            p  = pyaudio.PyAudio()
            lb = p.get_default_wasapi_loopback()
            print(f"  Default WASAPI loopback: {lb['name'] if lb else 'NONE'}")
            p.terminate()


if __name__ == "__main__":
    import sys
    if "--devices" in sys.argv:
        AudioInput.list_devices()
    elif len(sys.argv) > 1:
        ai = AudioInput()
        ai.register_callback(lambda s, r: None)
        ai.stream_file(sys.argv[1])
