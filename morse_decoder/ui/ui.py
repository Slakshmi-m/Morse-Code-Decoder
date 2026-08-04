"""
ui.py — Tkinter Live Decoder Dashboard
=======================================
Assignment requirement: "A dedicated window must display the decoded
text in real-time as it is being processed."

Layout
------
  Top bar      : control buttons (Open WAV, Microphone, System Audio, Stop, Clear)
  Left column  : five matplotlib panels (oscilloscope, FFT, waterfall,
                 binary signal, dit-dah histogram) — live-updating
  Right column : decoded-letter tiles (large coloured cards with Morse dots
                 beneath), Morse symbol stream, full decoded text box

The dashboard accumulates audio in a rolling buffer (BUFFER_SECONDS) and
re-decodes + re-draws every REFRESH_MS milliseconds using Tkinter's
after() scheduler — no threads touching the GUI directly.
"""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from scipy.signal import butter, lfilter, welch
    from scipy.signal import spectrogram as scipy_spectrogram
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

import threading

from dsp.audio_input import AudioInput
from dsp.constants   import MORSE_MAP, TEXT_TO_MORSE
from dsp.corrector   import MorseCorrector
from dsp.engine      import MorseEngine

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────
BUFFER_SECONDS = 6       # rolling audio window decoded each cycle
REFRESH_MS     = 250     # GUI update interval in ms
MAX_TILES      = 8       # letter tiles visible in the right panel
SAMPLE_RATE    = 8_000

# ─────────────────────────────────────────────────────────────────────────────
# Colour tokens (dark SDR theme)
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#0D0D1A"
PANEL_BG  = "#11111F"
ACCENT    = "#00FF88"
ORANGE    = "#FF8800"
YELLOW    = "#FFE500"
CYAN      = "#00CFFF"
RED_COL   = "#FF4466"
BLUE_COL  = "#4488FF"
GREY      = "#555577"
LABEL_COL = "#AAAACC"
WHITE     = "#E8E8FF"

TILE_COLOURS = ["#1A3A4A", "#1A4A3A", "#3A1A4A", "#4A3A1A",
                "#1A1A4A", "#4A1A1A", "#1A4A4A", "#4A1A4A"]


class DecoderUI:
    """All-in-one live Morse decoder dashboard."""

    def __init__(self, initial_file: Optional[str] = None) -> None:
        self._root = tk.Tk()
        self._root.title("Morse Code Decoder  ·  Live Dashboard")
        self._root.configure(bg=BG)
        self._root.minsize(1100, 700)

        # ── State ─────────────────────────────────────────────────────────────
        self._audio       = AudioInput(sample_rate=SAMPLE_RATE)
        self._audio._on_error = self._on_audio_error
        self._corrector   = MorseCorrector()
        self._buffer      = np.zeros(0, dtype=np.int16)
        self._buf_max     = BUFFER_SECONDS * SAMPLE_RATE
        self._chunk_q: queue.Queue[np.ndarray] = queue.Queue()
        self._full_text   = ""
        self._running     = False
        self._source_label= tk.StringVar(value="—")
        self._decode_mode    = tk.StringVar(value="DSP")
        self._ml_pending     = False
        self._file_mode      = False
        self._last_file_path: Optional[str] = None

        # Register audio callback (thread-safe via queue)
        self._audio.register_callback(self._on_audio_chunk)

        # ── Build UI ─────────────────────────────────────────────────────────
        self._build_toolbar()
        self._build_main_area()

        # ── Start refresh loop ────────────────────────────────────────────────
        self._root.after(REFRESH_MS, self._refresh)

        # ── Auto-open file if given ───────────────────────────────────────────
        if initial_file:
            self._root.after(200, lambda: self._open_file(initial_file))

    # ──────────────────────────────────────────────────────────────────────────
    # Toolbar
    # ──────────────────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self._root, bg=PANEL_BG, pady=6)
        bar.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(bar, text="Morse Code Decoder",
                         bg=PANEL_BG, fg=ACCENT,
                         font=("Consolas", 14, "bold"))
        title.pack(side=tk.LEFT, padx=14)

        sub = tk.Label(bar, text="Live Dashboard",
                       bg=PANEL_BG, fg=GREY, font=("Consolas", 9))
        sub.pack(side=tk.LEFT)

        btn_cfg = dict(bg="#1C2A3A", fg=WHITE, activebackground="#2A4A6A",
                       activeforeground=WHITE, relief=tk.FLAT,
                       font=("Consolas", 9), padx=10, pady=4, cursor="hand2")

        tk.Button(bar, text="⊙ Open Audio",
                  command=self._browse_wav, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="● Microphone",
                  command=self._start_mic, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="◉ System Audio",
                  command=self._start_system_audio, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="■ Stop",
                  command=self._stop, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="✕ Clear",
                  command=self._clear, **btn_cfg).pack(side=tk.LEFT, padx=4)

        src = tk.Label(bar, textvariable=self._source_label,
                       bg=PANEL_BG, fg=CYAN, font=("Consolas", 9))
        src.pack(side=tk.RIGHT, padx=14)

        # ML / DSP mode toggle
        mode_frame = tk.Frame(bar, bg=PANEL_BG)
        mode_frame.pack(side=tk.RIGHT, padx=10)
        tk.Label(mode_frame, text="Mode:", bg=PANEL_BG, fg=LABEL_COL,
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        for mode, col in [("DSP", ACCENT), ("ML", CYAN)]:
            tk.Radiobutton(mode_frame, text=mode, variable=self._decode_mode,
                           value=mode, bg=PANEL_BG, fg=col,
                           selectcolor=PANEL_BG, activebackground=PANEL_BG,
                           font=("Consolas", 8, "bold")
                           ).pack(side=tk.LEFT, padx=4)

    # ──────────────────────────────────────────────────────────────────────────
    # Main area: left = plots, right = decoded text
    # ──────────────────────────────────────────────────────────────────────────

    def _build_main_area(self) -> None:
        main = tk.Frame(self._root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Left: matplotlib canvas ───────────────────────────────────────────
        if _MPL_OK:
            self._fig, self._axes = self._create_figure()
            self._canvas = FigureCanvasTkAgg(self._fig, master=main)
            self._canvas.get_tk_widget().pack(
                side=tk.LEFT, fill=tk.BOTH, expand=True
            )
        else:
            tk.Label(main, text="matplotlib not installed\npip install matplotlib",
                     bg=BG, fg=RED_COL, font=("Consolas", 12)
                     ).pack(side=tk.LEFT, expand=True)

        # ── Right: decoded output panel ───────────────────────────────────────
        right = tk.Frame(main, bg=PANEL_BG, width=310)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=0)
        right.pack_propagate(False)

        self._build_right_panel(right)

    def _create_figure(self):
        """Create the 5-panel matplotlib figure."""
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(10, 8), facecolor=BG, tight_layout=False)
        fig.subplots_adjust(
            top=0.96, bottom=0.06,
            left=0.08, right=0.97,
            hspace=0.52, wspace=0.35,
        )
        gs = gridspec.GridSpec(3, 2, figure=fig,
                               height_ratios=[1, 1.1, 1])

        ax_wave  = fig.add_subplot(gs[0, 0])
        ax_fft   = fig.add_subplot(gs[0, 1])
        ax_wfall = fig.add_subplot(gs[1, :])
        ax_bin   = fig.add_subplot(gs[2, 0])
        ax_hist  = fig.add_subplot(gs[2, 1])

        axes = dict(wave=ax_wave, fft=ax_fft,
                    wfall=ax_wfall, bin=ax_bin, hist=ax_hist)
        for ax in axes.values():
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=LABEL_COL, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(GREY)

        # Static titles
        ax_wave.set_title("① Oscilloscope  (Raw Waveform)",
                          color=WHITE, fontsize=8, loc="left", pad=3)
        ax_fft.set_title("② FFT Spectrum  (Frequency Domain)",
                         color=WHITE, fontsize=8, loc="left", pad=3)
        ax_wfall.set_title("③ Waterfall / Spectrogram",
                           color=WHITE, fontsize=8, loc="left", pad=3)
        ax_bin.set_title("④ Binary Signal  (ON = Dit or Dah)",
                         color=WHITE, fontsize=8, loc="left", pad=3)
        ax_hist.set_title("⑤ Dit / Dah / Space Durations",
                          color=WHITE, fontsize=8, loc="left", pad=3)

        return fig, axes

    def _build_right_panel(self, parent: tk.Frame) -> None:
        """Build the decoded letters / text section."""
        pad = dict(padx=10, pady=4)

        # Section: decoded letter tiles
        tk.Label(parent, text="DECODED LETTERS",
                 bg=PANEL_BG, fg=LABEL_COL,
                 font=("Consolas", 8, "bold")).pack(anchor="w", **pad)

        self._tile_frame = tk.Frame(parent, bg=PANEL_BG)
        self._tile_frame.pack(fill=tk.X, **pad)
        self._tiles: list[dict] = []
        for i in range(MAX_TILES):
            col = TILE_COLOURS[i % len(TILE_COLOURS)]
            frm = tk.Frame(self._tile_frame, bg=col, width=32, height=52)
            frm.pack(side=tk.LEFT, padx=2)
            frm.pack_propagate(False)
            ltr = tk.Label(frm, text="", bg=col, fg=WHITE,
                           font=("Consolas", 18, "bold"))
            ltr.pack(expand=True)
            sub = tk.Label(frm, text="", bg=col, fg=LABEL_COL,
                           font=("Consolas", 6))
            sub.pack()
            self._tiles.append({"frame": frm, "letter": ltr, "morse": sub, "bg": col})

        tk.Frame(parent, bg=GREY, height=1).pack(fill=tk.X, padx=10, pady=6)

        # Section: raw Morse symbols
        tk.Label(parent, text="MORSE SYMBOLS",
                 bg=PANEL_BG, fg=LABEL_COL,
                 font=("Consolas", 8, "bold")).pack(anchor="w", **pad)
        self._morse_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._morse_var,
                 bg=PANEL_BG, fg=YELLOW,
                 font=("Consolas", 10), wraplength=280, justify=tk.LEFT
                 ).pack(anchor="w", **pad)

        tk.Frame(parent, bg=GREY, height=1).pack(fill=tk.X, padx=10, pady=6)

        # Section: full decoded text
        tk.Label(parent, text="FULL DECODED TEXT",
                 bg=PANEL_BG, fg=LABEL_COL,
                 font=("Consolas", 8, "bold")).pack(anchor="w", **pad)
        self._text_box = scrolledtext.ScrolledText(
            parent, bg="#0A0A18", fg=WHITE,
            font=("Consolas", 12), wrap=tk.WORD,
            relief=tk.FLAT, height=10, state=tk.DISABLED,
        )
        self._text_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # SNR badge
        self._snr_var = tk.StringVar(value="SNR: —")
        tk.Label(parent, textvariable=self._snr_var,
                 bg=PANEL_BG, fg=CYAN, font=("Consolas", 8)
                 ).pack(anchor="e", padx=10, pady=2)

    # ──────────────────────────────────────────────────────────────────────────
    # Audio control buttons
    # ──────────────────────────────────────────────────────────────────────────

    def _browse_wav(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3"),
                ("WAV files",   "*.wav"),
                ("MP3 files",   "*.mp3"),
                ("All files",   "*.*"),
            ],
        )
        if path:
            self._open_file(path)

    def _open_file(self, path: str) -> None:
        self._stop()
        self._clear()
        self._file_mode      = True
        self._last_file_path = path
        ext = path.rsplit(".", 1)[-1].upper() if "." in path else "FILE"
        self._source_label.set(f"▶ {ext}: {path.replace('\\', '/').split('/')[-1]}")
        self._running = True
        self._audio._on_complete = lambda: self._root.after(0, self._on_audio_file_done)
        self._audio.stream_file_async(path, realtime=True)

    def _start_mic(self) -> None:
        self._stop()
        self._clear()
        self._file_mode = False
        try:
            self._audio.start_microphone()
            self._running = True
            self._source_label.set("● Microphone")
        except RuntimeError as exc:
            messagebox.showerror("Microphone Error", str(exc))

    def _start_system_audio(self) -> None:
        self._stop()
        self._clear()
        self._file_mode = False
        try:
            label = self._audio.start_system_audio()
            self._running = True
            self._source_label.set(f"◉ {label}")
        except RuntimeError as exc:
            if "NEED_PAWPATCH" in str(exc):
                messagebox.showinfo(
                    "Install required",
                    "System audio capture needs pyaudiowpatch.\n\n"
                    "Run in your terminal:\n    pip install pyaudiowpatch",
                )
            else:
                messagebox.showerror("System Audio Error", str(exc))

    def _stop(self) -> None:
        self._running   = False
        self._file_mode = False
        self._audio.stop()
        self._source_label.set("■ Stopped")

    def _clear(self) -> None:
        self._buffer    = np.zeros(0, dtype=np.int16)
        self._full_text = ""
        self._morse_var.set("")
        self._snr_var.set("SNR: —")
        # Clear tiles
        for t in self._tiles:
            t["letter"]["text"] = ""
            t["morse"]["text"]  = ""
        # Clear text box
        self._text_box.configure(state=tk.NORMAL)
        self._text_box.delete("1.0", tk.END)
        self._text_box.configure(state=tk.DISABLED)
        # Clear plots
        if _MPL_OK:
            for ax in self._axes.values():
                ax.cla()
                ax.set_facecolor(PANEL_BG)
            self._canvas.draw_idle()

    # ──────────────────────────────────────────────────────────────────────────
    # Audio callback (called from capture thread → push to queue)
    # ──────────────────────────────────────────────────────────────────────────

    def _on_audio_chunk(self, samples: np.ndarray, _sr: int) -> None:
        self._chunk_q.put(samples)

    def _on_audio_error(self, msg: str) -> None:
        self._root.after(0, lambda: messagebox.showerror("Audio Error", msg))

    # ──────────────────────────────────────────────────────────────────────────
    # Refresh loop — runs in the Tk main thread via after()
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        new_data = False
        while not self._chunk_q.empty():
            chunk = self._chunk_q.get_nowait()
            self._buffer = np.concatenate([self._buffer, chunk])
            new_data = True

        if len(self._buffer) > self._buf_max:
            self._buffer = self._buffer[-self._buf_max:]

        if new_data and len(self._buffer) > SAMPLE_RATE // 2:
            self._decode_and_update()

        self._root.after(REFRESH_MS, self._refresh)

    # ──────────────────────────────────────────────────────────────────────────
    # Decode + UI update
    # ──────────────────────────────────────────────────────────────────────────

    def _decode_and_update(self) -> None:
        data = self._buffer.copy()

        if self._decode_mode.get() == "ML":
            # Always decode with DSP so tiles/text are never blank during streaming.
            # ML result overwrites when it finishes (file: end of file; live: each cycle).
            try:
                engine   = MorseEngine(SAMPLE_RATE, data)
                dsp_text = engine.decode()
                self._snr_var.set(f"SNR: {engine.snr_db:.1f} dB")
                if dsp_text and not dsp_text.startswith("["):
                    self._update_right_panel(dsp_text)
                if _MPL_OK:
                    self._update_plots(data, engine)
            except Exception:
                pass
            # Run ML inference only for live sources (file ML runs at end-of-file)
            if not self._file_mode and not self._ml_pending:
                self._ml_pending = True
                threading.Thread(target=self._run_ml_inference,
                                 args=(data,), daemon=True).start()
            return

        try:
            engine   = MorseEngine(SAMPLE_RATE, data)
            raw_text = engine.decode()
            snr      = engine.snr_db
        except Exception as exc:
            print(f"[UI] Engine error: {exc}")
            return

        if raw_text and not raw_text.startswith("["):
            corrected = self._corrector.correct(raw_text)
        else:
            corrected = raw_text

        self._snr_var.set(f"SNR: {snr:.1f} dB")
        self._update_right_panel(corrected)
        if _MPL_OK:
            self._update_plots(data, engine)

    def _run_ml_inference(self, data: np.ndarray) -> None:
        try:
            from ml.inference import decode_buffer_ml
            result = decode_buffer_ml(data, sr=SAMPLE_RATE)
        except Exception as exc:
            result = f"[ML error: {exc}]"
        finally:
            self._ml_pending = False
        if result and not result.startswith("["):
            self._root.after(0, lambda r=result: self._update_right_panel(r))
        elif result and result.startswith("["):
            self._root.after(0, lambda r=result: self._snr_var.set(r[:60]))

    def _on_audio_file_done(self) -> None:
        lbl = self._source_label.get()
        if lbl.startswith("▶ "):
            self._source_label.set("✓ " + lbl[2:])
        self._file_mode = False
        if self._decode_mode.get() == "ML" and not self._ml_pending:
            self._ml_pending = True
            path = self._last_file_path
            if path:
                threading.Thread(target=self._run_ml_file_inference,
                                 args=(path,), daemon=True).start()
            elif len(self._buffer) > 0:
                data = self._buffer.copy()
                threading.Thread(target=self._run_ml_inference,
                                 args=(data,), daemon=True).start()

    def _run_ml_file_inference(self, path: str) -> None:
        """Run ML on the complete audio file (WAV or MP3)."""
        try:
            if path.lower().endswith(".mp3"):
                from ml.inference import decode_buffer_ml
                from dsp.audio_input import AudioInput
                samples, fs = AudioInput._load_mp3(path)
                if fs != SAMPLE_RATE:
                    import numpy as _np
                    n       = int(len(samples) * SAMPLE_RATE / fs)
                    samples = _np.interp(
                        _np.linspace(0, 1, n),
                        _np.linspace(0, 1, len(samples)),
                        samples.astype(_np.float32),
                    ).clip(-32767, 32767).astype(_np.int16)
                result = decode_buffer_ml(samples, sr=SAMPLE_RATE)
            else:
                from ml.inference import decode_wav_ml
                result = decode_wav_ml(path, sr=SAMPLE_RATE)
        except Exception as exc:
            result = f"[ML error: {exc}]"
        finally:
            self._ml_pending = False
        if result and not result.startswith("["):
            self._root.after(0, lambda r=result: self._update_right_panel(r))
        elif result and result.startswith("["):
            self._root.after(0, lambda r=result: self._snr_var.set(r[:60]))

    def _update_right_panel(self, text: str) -> None:
        """Update letter tiles, Morse symbols, and full text box."""
        if not text or text.startswith("["):
            return

        # Accumulate only new printable characters
        new_chars = [c for c in text if c.strip()]
        if not new_chars:
            return

        # Build Morse symbol string for display
        morse_parts = []
        for ch in text:
            if ch == " ":
                morse_parts.append("/")
            else:
                morse_parts.append(TEXT_TO_MORSE.get(ch.upper(), "?"))
        self._morse_var.set("  ".join(morse_parts[-12:]))

        # Update letter tiles (last MAX_TILES decoded chars)
        display_chars = [c for c in text if c.strip()][-MAX_TILES:]
        for i, td in enumerate(self._tiles):
            if i < len(display_chars):
                ch = display_chars[i].upper()
                td["letter"]["text"] = ch
                td["morse"]["text"]  = TEXT_TO_MORSE.get(ch, "")
            else:
                td["letter"]["text"] = ""
                td["morse"]["text"]  = ""

        # Accumulate decoded text: find overlap between what was shown before
        # and what the new decode produced, then extend rather than replace.
        new_text = text.strip()
        if not new_text:
            return
        prev = self._full_text.strip()
        if not prev:
            self._full_text = new_text
        elif new_text.startswith(prev):
            # New result is a direct extension of previous — keep growing
            self._full_text = new_text
        elif prev.endswith(new_text) or prev == new_text:
            # No new content yet — skip redraw
            return
        else:
            # Find longest suffix of prev that is a prefix of new_text
            max_ov = min(len(prev), len(new_text))
            overlap = 0
            for k in range(max_ov, 0, -1):
                if prev[-k:] == new_text[:k]:
                    overlap = k
                    break
            if overlap > 0:
                self._full_text = prev + new_text[overlap:]
            else:
                # Completely new content (e.g. new file) — append with separator
                self._full_text = (prev + "  " + new_text) if prev else new_text

        self._text_box.configure(state=tk.NORMAL)
        self._text_box.delete("1.0", tk.END)
        self._text_box.insert(tk.END, self._full_text)
        self._text_box.see(tk.END)
        self._text_box.configure(state=tk.DISABLED)

    # ──────────────────────────────────────────────────────────────────────────
    # Live plot update
    # ──────────────────────────────────────────────────────────────────────────

    def _update_plots(self, data: np.ndarray, engine: MorseEngine) -> None:
        """Redraw all five matplotlib panels with fresh data."""
        # Normalise for display
        raw_f = data.astype(np.float32)
        peak  = np.max(np.abs(raw_f)) or 1.0
        norm  = raw_f / peak
        disp  = norm[:int(BUFFER_SECONDS * SAMPLE_RATE)]
        sr    = SAMPLE_RATE

        # ── Panel 1: Oscilloscope ─────────────────────────────────────────────
        ax = self._axes["wave"]
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        t = np.linspace(0, len(disp) / sr, len(disp))
        ax.plot(t, disp, color=ACCENT, linewidth=0.35, alpha=0.9)
        ax.set_title("① Oscilloscope  (Raw Waveform)",
                     color=WHITE, fontsize=8, loc="left", pad=3)
        ax.set_xlim(0, t[-1]); ax.set_ylim(-1.15, 1.15)
        ax.tick_params(colors=LABEL_COL, labelsize=7)
        ax.set_xlabel("Time (s)", color=LABEL_COL, fontsize=7)
        ax.grid(True, color=GREY, linewidth=0.3, alpha=0.4)
        for sp in ax.spines.values(): sp.set_color(GREY)

        # ── Panel 2: FFT Spectrum ─────────────────────────────────────────────
        ax = self._axes["fft"]
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        win  = np.hanning(len(disp))
        mags = np.abs(np.fft.rfft(disp * win))
        freq = np.fft.rfftfreq(len(disp), d=1.0 / sr)
        mask = (freq >= 200) & (freq <= 1500)
        f_, m_ = freq[mask], mags[mask]
        ax.fill_between(f_, m_, color=ORANGE, alpha=0.45, linewidth=0)
        ax.plot(f_, m_, color=ORANGE, linewidth=0.8)
        if len(m_):
            pk = f_[np.argmax(m_)]
            ax.axvline(pk, color=YELLOW, linewidth=1.2, linestyle="--", alpha=0.9)
            ax.text(pk + 20, m_[np.argmax(m_)] * 0.8,
                    f"{pk:.0f} Hz", color=YELLOW, fontsize=7)
        ax.set_title("② FFT Spectrum  (Frequency Domain)",
                     color=WHITE, fontsize=8, loc="left", pad=3)
        ax.set_xlim(200, 1500)
        ax.tick_params(colors=LABEL_COL, labelsize=7)
        ax.set_xlabel("Frequency (Hz)", color=LABEL_COL, fontsize=7)
        ax.grid(True, color=GREY, linewidth=0.3, alpha=0.4)
        for sp in ax.spines.values(): sp.set_color(GREY)

        # ── Panel 3: Waterfall ────────────────────────────────────────────────
        ax = self._axes["wfall"]
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        nperseg  = min(256, max(32, len(disp) // 60))
        noverlap = nperseg * 3 // 4
        try:
            f_, t_, Sxx = scipy_spectrogram(disp, fs=sr,
                                            nperseg=nperseg, noverlap=noverlap)
            Sxx_db = 10 * np.log10(Sxx + 1e-12)
            fm = (f_ >= 200) & (f_ <= 1500)
            if np.any(fm) and len(t_) > 1:
                z    = Sxx_db[fm]
                vmin = float(np.percentile(z, 8))
                vmax = float(np.percentile(z, 99))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                ax.pcolormesh(f_[fm], t_, z.T,
                              shading="auto", cmap="inferno",
                              vmin=vmin, vmax=vmax)
                cf = engine.detected_freq
                ax.axvline(cf, color=YELLOW, linewidth=1.0, linestyle="--", alpha=0.8)
                ax.text(cf + 15, t_[-1] * 0.01,
                        f"{cf:.0f} Hz", color=YELLOW, fontsize=7, ha="left", va="top")
        except Exception as exc:
            print(f"[UI] Waterfall error: {exc}")
        ax.set_title("③ Waterfall / Spectrogram  —  Morse = bright stripe",
                     color=WHITE, fontsize=8, loc="left", pad=3)
        ax.tick_params(colors=LABEL_COL, labelsize=7)
        ax.set_xlabel("Frequency (Hz)", color=LABEL_COL, fontsize=7)
        ax.set_ylabel("Time (s)",       color=LABEL_COL, fontsize=7)
        for sp in ax.spines.values(): sp.set_color(GREY)

        # ── Panel 4: Binary Signal ────────────────────────────────────────────
        ax = self._axes["bin"]
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        env    = engine.data
        win    = max(1, int(sr * 0.005))
        smooth = np.convolve(env, np.ones(win) / win, mode="same")
        lo     = np.percentile(smooth, 5)
        hi     = np.percentile(smooth, 95)
        binary = (smooth > lo + (hi - lo) * 0.45).astype(np.int8)
        tb     = np.linspace(0, len(binary) / sr, len(binary))
        ax.fill_between(tb, binary.astype(float),
                        step="pre", color=CYAN, alpha=0.75, linewidth=0)
        ax.step(tb, binary.astype(float),
                color=CYAN, linewidth=0.6, where="pre")
        ax.set_title("④ Binary Signal  (ON = Dit or Dah)",
                     color=WHITE, fontsize=8, loc="left", pad=3)
        ax.set_xlim(0, tb[-1]); ax.set_ylim(-0.15, 1.25)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["OFF", "ON"], color=LABEL_COL, fontsize=7)
        ax.tick_params(colors=LABEL_COL, labelsize=7)
        ax.grid(True, axis="x", color=GREY, linewidth=0.3, alpha=0.4)
        for sp in ax.spines.values(): sp.set_color(GREY)

        # ── Panel 5: Dit / Dah histogram ─────────────────────────────────────
        ax = self._axes["hist"]
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        changes = np.diff(binary.astype(int))
        idx     = np.concatenate(([0], np.where(changes != 0)[0] + 1, [len(binary)]))
        segs    = [(int(binary[idx[i]]), int(idx[i+1] - idx[i]))
                   for i in range(len(idx) - 1)]
        on_durs = [d for s, d in segs if s == 1 and d > 0]
        if on_durs:
            on_arr = np.array(sorted(on_durs))
            med    = float(np.median(on_arr))
            dits_s = [d for d in on_durs if d < med]
            unit   = float(np.mean(dits_s)) if dits_s else med / 2.0
            ms     = 1000.0 / sr
            dits   = [d * ms for s, d in segs if s == 1 and d < unit * 2]
            dahs   = [d * ms for s, d in segs if s == 1 and d >= unit * 2]
            spaces = [d * ms for s, d in segs if s == 0 and d >= unit * 0.5]
            all_v  = dits + dahs + spaces
            if all_v:
                bins = np.linspace(0, min(max(all_v), unit * 12 * ms), 30)
                if dits:
                    ax.hist(dits,   bins=bins, color=ACCENT,    alpha=0.80,
                            label=f"Dits ({len(dits)})",    edgecolor="none")
                if dahs:
                    ax.hist(dahs,   bins=bins, color=RED_COL,   alpha=0.80,
                            label=f"Dahs ({len(dahs)})",    edgecolor="none")
                if spaces:
                    ax.hist(spaces, bins=bins, color=BLUE_COL,  alpha=0.55,
                            label=f"Spaces ({len(spaces)})", edgecolor="none")
                ax.axvline(unit * 2 * ms, color=YELLOW,
                           linewidth=1.1, linestyle="--", alpha=0.9)
                ax.legend(fontsize=7, facecolor="#1A1A2E",
                          edgecolor=GREY, labelcolor=WHITE, loc="upper right")
        ax.set_title("⑤ Dit / Dah / Space Durations",
                     color=WHITE, fontsize=8, loc="left", pad=3)
        ax.tick_params(colors=LABEL_COL, labelsize=7)
        ax.set_xlabel("Duration (ms)", color=LABEL_COL, fontsize=7)
        ax.set_ylabel("Count",         color=LABEL_COL, fontsize=7)
        ax.grid(True, axis="y", color=GREY, linewidth=0.3, alpha=0.4)
        for sp in ax.spines.values(): sp.set_color(GREY)

        self._canvas.draw_idle()

    # ──────────────────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.mainloop()

    def _on_close(self) -> None:
        self._stop()
        self._root.destroy()
