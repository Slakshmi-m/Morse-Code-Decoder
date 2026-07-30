"""
audio_input.py — Audio Input (WAV file / Microphone / System Audio)
=====================================================================
System audio capture uses pyaudiowpatch which supports true WASAPI
loopback on Windows — this works even when Stereo Mix gives WDM-KS
errors with sounddevice.

INSTALL (run once in your terminal):
    pip install sounddevice pyaudiowpatch

pyaudiowpatch is a fork of PyAudio that adds native Windows WASAPI
loopback support.  It finds your default playback device (speakers /
headphones) and captures exactly what is playing through it —
YouTube, Spotify, any browser audio — without needing Stereo Mix at all.
"""

import threading, time
from typing import Callable, List, Optional
import numpy as np
from . import load_audio_file

# ── sounddevice (mic + WAV) ───────────────────────────────────────────────────
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

# ── pyaudiowpatch (WASAPI loopback for system audio) ─────────────────────────
try:
    import pyaudiowpatch as pyaudio
    _PAW_AVAILABLE = True
except ImportError:
    _PAW_AVAILABLE = False

ChunkCallback = Callable[[np.ndarray, int], None]

_LOOPBACK_KEYWORDS = [
    "stereo mix", "what u hear", "wave out mix",
    "cable output", "vb-audio", "virtual audio cable",
    "loopback", "mix output", "blackhole", "soundflower",
]


# ═════════════════════════════════════════════════════════════════════════════
class AudioInput:
    SAMPLE_RATE    = 8000
    CHUNK_DURATION = 0.1          # seconds per chunk

    def __init__(self, sample_rate=SAMPLE_RATE,
                 chunk_duration=CHUNK_DURATION, device=None):
        self.sample_rate    = sample_rate
        self.chunk_duration = chunk_duration
        self.device         = device
        self.chunk_size     = int(sample_rate * chunk_duration)
        self._callbacks: List[ChunkCallback] = []
        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._on_error  = None    # set by ui.py to show popups

    def register_callback(self, fn: ChunkCallback):
        self._callbacks.append(fn)

    def _dispatch(self, samples: np.ndarray):
        for cb in self._callbacks:
            try:
                cb(samples, self.sample_rate)
            except Exception as e:
                print(f"[AudioInput] Callback error: {e}")

    # ── microphone via sounddevice ────────────────────────────────────────────

    def start_microphone(self, device=None):
        if not _SD_AVAILABLE:
            raise RuntimeError("Run: pip install sounddevice")
        dev = device if device is not None else self.device
        self._running = True
        self._thread  = threading.Thread(
            target=self._sd_stream_loop, args=(dev,),
            daemon=True, name="mic")
        self._thread.start()
        name = sd.query_devices(dev)["name"] if dev is not None \
               else sd.query_devices(kind="input")["name"]
        print(f"[AudioInput] Mic: {name}")

    def _sd_stream_loop(self, device):
        """sounddevice capture loop — used for microphone."""
        try:
            info      = sd.query_devices(device) if device is not None \
                        else sd.query_devices(kind="input")
            native_sr = int(info["default_samplerate"])
            channels  = min(2, max(1, int(info["max_input_channels"])))
            blk       = int(native_sr * self.chunk_duration)

            with sd.InputStream(samplerate=native_sr, channels=channels,
                                dtype="float32", device=device,
                                blocksize=blk, latency="low") as stream:
                print(f"[AudioInput] Mic stream open ✓")
                while self._running:
                    chunk, _ = stream.read(blk)
                    mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.ravel()
                    if native_sr != self.sample_rate:
                        mono = self._resample(mono, native_sr, self.sample_rate)
                    self._dispatch((mono * 32767).clip(-32767,32767).astype(np.int16))
        except Exception as e:
            print(f"[AudioInput] Mic error: {e}")
            self._running = False

    # ── system audio via pyaudiowpatch (WASAPI loopback) ─────────────────────

    def start_system_audio(self) -> str:
        """
        Capture everything playing on the PC speakers/headphones using
        WASAPI loopback via pyaudiowpatch.

        This works WITHOUT Stereo Mix and avoids the WDM-KS -9999 error.
        Raises RuntimeError if pyaudiowpatch is not installed.
        """
        if not _PAW_AVAILABLE:
            raise RuntimeError("NEED_PAWPATCH")   # ui.py shows install instructions

        self._running = True
        self._thread  = threading.Thread(
            target=self._wasapi_loop,
            daemon=True, name="wasapi-loopback")
        self._thread.start()
        return "WASAPI Loopback"

    def _wasapi_loop(self):
        """
        WASAPI loopback capture using pyaudiowpatch.
        Automatically finds the default output device (speakers/headphones)
        and records its loopback stream — exactly what you hear.
        """
        try:
            p = pyaudio.PyAudio()

            # Find the default output (playback) device
            default_speakers = p.get_default_wasapi_loopback()
            if default_speakers is None:
                raise RuntimeError("No WASAPI loopback device found")

            native_sr = int(default_speakers["defaultSampleRate"])
            channels  = min(2, int(default_speakers["maxInputChannels"]))
            dev_idx   = default_speakers["index"]

            print(f"[AudioInput] WASAPI loopback: {default_speakers['name']!r} "
                  f"sr={native_sr} ch={channels}")

            chunk_frames = int(native_sr * self.chunk_duration)

            stream = p.open(
                format              = pyaudio.paFloat32,
                channels            = channels,
                rate                = native_sr,
                input               = True,
                input_device_index  = dev_idx,
                frames_per_buffer   = chunk_frames,
            )

            print("[AudioInput] WASAPI loopback stream open ✓")

            while self._running:
                raw = stream.read(chunk_frames, exception_on_overflow=False)
                arr = np.frombuffer(raw, dtype=np.float32).copy()

                # stereo → mono
                if channels == 2:
                    arr = arr.reshape(-1, 2).mean(axis=1)

                # resample
                if native_sr != self.sample_rate:
                    arr = self._resample(arr, native_sr, self.sample_rate)

                self._dispatch((arr * 32767).clip(-32767,32767).astype(np.int16))

            stream.stop_stream()
            stream.close()
            p.terminate()

        except Exception as e:
            print(f"[AudioInput] WASAPI loop error: {e}")
            self._running = False
            if callable(self._on_error):
                self._on_error(str(e))

    # ── Audio file (WAV / MP3) ────────────────────────────────────────────────

    def stream_file(self, path: str, realtime=False):
        try:
            fs, raw = load_audio_file(path)
        except Exception as e:
            msg = f"Cannot read audio file: {e}"
            print(f"[AudioInput] {msg}")
            if self._on_error:
                self._on_error(msg)
            return

        if raw.ndim > 1:
            raw = raw[:, 0]
        if fs != self.sample_rate:
            raw = self._resample(raw.astype(np.float32),
                                 fs, self.sample_rate).astype(np.int16)

        print(f"[AudioInput] Streaming '{path}' ({len(raw)/self.sample_rate:.1f}s)")
        self._running = True
        offset = 0
        while self._running and offset < len(raw):
            chunk = raw[offset: offset + self.chunk_size]
            if not len(chunk): break
            self._dispatch(chunk)
            offset += self.chunk_size
            if realtime:
                time.sleep(self.chunk_duration)
        self._running = False

    def stream_file_async(self, path: str, realtime=True):
        self._thread = threading.Thread(
            target=self.stream_file, args=(path, realtime),
            daemon=True, name="file")
        self._thread.start()

    # ── stop ──────────────────────────────────────────────────────────────────

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resample(data, orig, target):
        if orig == target: return data
        n = int(len(data) * target / orig)
        return np.interp(np.linspace(0,1,n),
                         np.linspace(0,1,len(data)), data).astype(data.dtype)

    @staticmethod
    def list_devices():
        print("\n=== sounddevice INPUT devices ===")
        if _SD_AVAILABLE:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    tag = " ← LOOPBACK" if any(
                        kw in d["name"].lower() for kw in _LOOPBACK_KEYWORDS) else ""
                    print(f"  [{i:2d}] {d['name']}{tag}")
        else:
            print("  sounddevice not installed")

        print(f"\npyaudiowpatch available: {_PAW_AVAILABLE}")
        if _PAW_AVAILABLE:
            p = pyaudio.PyAudio()
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
