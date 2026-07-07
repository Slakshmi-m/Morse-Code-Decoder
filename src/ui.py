"""
ui.py — Morse Code Decoder  ·  Modern Live Dashboard
=====================================================
Complete redesign. One window, everything live.

LEFT  (60%)  — 5 signal panels updating in real-time
RIGHT (40%)  — live decoded output with three views:
                 • Letter tiles  (big coloured boxes, one per character)
                 • Morse symbols (dots & dashes that built each letter)
                 • Full text     (scrolling plain-text transcript)

New in this version
-------------------
  • Modern card-based toolbar with hover effects
  • Live letter tiles — each decoded character appears as a glowing card
    showing both the letter AND its Morse code underneath
  • SNR meter bar (green → amber → red) in the status strip
  • Signal quality badge (GOOD / OK / WEAK) next to the source name
  • Carrier frequency and WPM shown as live badge pills
  • Cleaner colour system — Catppuccin Mocha palette
"""

import os, sys, queue, threading, time
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, lfilter, spectrogram as scipy_spectrogram, welch

# ── Catppuccin Mocha palette ──────────────────────────────────────────────────
BG        = "#1E1E2E"   # base
BG2       = "#181825"   # mantle
BG3       = "#11111B"   # crust
SURFACE0  = "#313244"
SURFACE1  = "#45475A"
OVERLAY   = "#6C7086"
C_WHITE   = "#CDD6F4"   # text
C_SUBTEXT = "#A6ADC8"
C_GREEN   = "#A6E3A1"
C_TEAL    = "#94E2D5"
C_BLUE    = "#89B4FA"
C_LAVENDER= "#B4BEFE"
C_MAUVE   = "#CBA6F7"
C_PINK    = "#F5C2E7"
C_RED     = "#F38BA8"
C_YELLOW  = "#F9E2AF"
C_PEACH   = "#FAB387"
C_CYAN    = "#00CFFF"

# plot internals
PANEL_BG  = "#181825"
C_GREY    = "#45475A"
C_LABEL   = "#A6ADC8"

# Morse alphabet for tile display
_MORSE = {
    'A':'.-','B':'-...','C':'-.-.','D':'-..','E':'.','F':'..-.','G':'--.','H':'....',
    'I':'..','J':'.---','K':'-.-','L':'.-..','M':'--','N':'-.','O':'---','P':'.--.',
    'Q':'--.-','R':'.-.','S':'...','T':'-','U':'..-','V':'...-','W':'.--','X':'-..-',
    'Y':'-.--','Z':'--..','0':'-----','1':'.----','2':'..---','3':'...--','4':'....-',
    '5':'.....','6':'-....','7':'--...','8':'---..','9':'----.','.':'.-.-.-',
    ',':'--..--','?':'..--..','=':'-...-','/':'-..-.','@':'.--.-.', ' ': '   '
}

# One colour per letter tile (cycles through accent colours)
_TILE_COLOURS = [
    C_BLUE, C_MAUVE, C_GREEN, C_PEACH, C_TEAL,
    C_LAVENDER, C_PINK, C_YELLOW, C_RED, C_CYAN,
]


# ══════════════════════════════════════════════════════════════════════════════
# Signal DSP helpers
# ══════════════════════════════════════════════════════════════════════════════

def _bandpass(data: np.ndarray, sr: int):
    f, psd = welch(data, sr, nperseg=min(1024, len(data)))
    mask = (f > 200) & (f < 1400)
    carrier = float(f[mask][np.argmax(psd[mask])]) if np.any(mask) else 700.0
    lo = max(50.0, carrier - 200)
    hi = min(sr / 2 - 1, carrier + 200)
    nyq = sr / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return lfilter(b, a, data), carrier

def _envelope(filtered, sr):
    win = max(1, int(sr * 0.006))
    return np.convolve(np.abs(filtered), np.ones(win) / win, mode="same")

def _binary(env, sr=8000):
    lo = np.percentile(env, 5); hi = np.percentile(env, 95)
    thresh = lo + (hi - lo) * 0.45
    binary = (env > thresh).astype(np.int8)
    tps = np.sum(np.abs(np.diff(binary.astype(int)))) / (len(env) / sr)
    return np.zeros(len(env), dtype=np.int8) if tps > 150 else binary

def _segments(binary):
    changes = np.diff(binary.astype(int))
    idx = [0] + list(np.where(changes != 0)[0] + 1) + [len(binary)]
    return [(int(binary[idx[i]]), idx[i+1]-idx[i]) for i in range(len(idx)-1)]

def _unit(segs, sr):
    on = [d for s, d in segs if s == 1]
    if not on: return sr * 0.08
    med = float(np.median(on))
    dits = [d for d in on if d < med]
    return float(np.mean(dits)) if dits else med / 2.0

def _peak_freq(data, sr):
    f, psd = welch(data, sr, nperseg=min(1024, len(data)))
    mask = (f > 200) & (f < 1400)
    return float(f[mask][np.argmax(psd[mask])]) if np.any(mask) else 700.0

def _wpm(unit_samples, sr):
    dot_s = unit_samples / sr
    return round(1.2 / dot_s) if dot_s > 0 else 0


# ══════════════════════════════════════════════════════════════════════════════
# StreamingDecoder
# ══════════════════════════════════════════════════════════════════════════════

class StreamingDecoder:
    DECODE_WINDOW_SEC = 6.0
    MIN_SEC           = 1.5
    MAX_SEC           = 30.0
    DECODE_EVERY_SEC  = 0.4

    def __init__(self, sr, on_text, on_status, on_signal):
        self.sr        = sr
        self.on_text   = on_text
        self.on_status = on_status
        self.on_signal = on_signal
        self._buf      = np.array([], dtype=np.float32)
        self._last     = ""
        self._max      = int(self.MAX_SEC * sr)
        self._last_t   = 0.0
        from src.engine import MorseEngine
        self._Engine   = MorseEngine

    def push(self, samples, rate):
        self._buf = np.concatenate([self._buf, samples.astype(np.float32)])
        if len(self._buf) > self._max:
            self._buf = self._buf[-self._max:]
        self.on_signal(self._buf.copy())

        now = time.time()
        if now - self._last_t < self.DECODE_EVERY_SEC: return
        if len(self._buf) < int(self.MIN_SEC * self.sr):
            self.on_status(("buffering", 0, 0, 0)); return
        self._last_t = now

        win    = int(self.DECODE_WINDOW_SEC * self.sr)
        dbuf   = self._buf[-win:].copy().astype(np.int16)
        try:
            eng  = self._Engine(self.sr, dbuf)
            text = eng.decode()
            if text != self._last:
                self.on_text(text)
                self._last = text

            freq = _peak_freq(self._buf[-int(2*self.sr):], self.sr)
            filt, _ = _bandpass(dbuf.astype(np.float32), self.sr)
            env  = _envelope(filt, self.sr)
            segs = _segments(_binary(env, self.sr))
            u    = _unit(segs, self.sr)
            self.on_status(("live", eng.snr_db, freq, _wpm(u, self.sr)))
        except Exception as e:
            self.on_status(("error", 0, 0, 0))

    def reset(self):
        self._buf  = np.array([], dtype=np.float32)
        self._last = ""
        self._last_t = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LivePlotter — 5 panels
# ══════════════════════════════════════════════════════════════════════════════

class LivePlotter:
    REFRESH_MS      = 150
    WATERFALL_EVERY = 5
    FREQ_MIN        = 200
    FREQ_MAX        = 1400
    HISTORY_S       = 6.0

    def __init__(self, parent, sr):
        self.sr       = sr
        self._buf     = None
        self._running = False
        self._wf_tick = 0

        plt.style.use("dark_background")
        self.fig = plt.Figure(figsize=(8, 9), facecolor=BG, tight_layout=False)
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.96,
                                 bottom=0.05, hspace=0.58, wspace=0.32)
        gs = gridspec.GridSpec(3, 2, figure=self.fig,
                               height_ratios=[1, 1.15, 1],
                               hspace=0.58, wspace=0.32,
                               top=0.96, bottom=0.05,
                               left=0.08, right=0.97)
        self.ax_wave = self.fig.add_subplot(gs[0, 0])
        self.ax_fft  = self.fig.add_subplot(gs[0, 1])
        self.ax_fall = self.fig.add_subplot(gs[1, :])
        self.ax_bin  = self.fig.add_subplot(gs[2, 0])
        self.ax_hist = self.fig.add_subplot(gs[2, 1])

        self._style_axes()
        self._init_panels()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _style_axes(self):
        for ax in (self.ax_wave, self.ax_fft, self.ax_fall,
                   self.ax_bin, self.ax_hist):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=C_LABEL, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(C_GREY)

    def _init_panels(self):
        titles = [
            (self.ax_wave, "① Oscilloscope  (Raw Waveform)"),
            (self.ax_fft,  "② FFT Spectrum  (Frequency Domain)"),
            (self.ax_fall, "③ Waterfall / Spectrogram"),
            (self.ax_bin,  "④ Binary Signal  (ON = Dit or Dah)"),
            (self.ax_hist, "⑤ Dit / Dah / Space Durations"),
        ]
        for ax, t in titles:
            ax.cla()
            ax.set_facecolor(PANEL_BG)
            for sp in ax.spines.values(): sp.set_color(C_GREY)
            ax.set_title(t, color=C_WHITE, fontsize=8, pad=3, loc="left")
            ax.tick_params(colors=C_LABEL, labelsize=7)
            ax.text(0.5, 0.5, "Waiting for audio…",
                    transform=ax.transAxes, ha="center", va="center",
                    color=C_GREY, fontsize=9, style="italic")

    def push(self, buf): self._buf = buf

    def start(self, root):
        self._root = root; self._running = True; self._schedule()

    def stop(self): self._running = False

    def reset(self):
        self._buf = None; self._init_panels(); self.canvas.draw_idle()

    def _schedule(self):
        if self._running:
            self._redraw()
            self._root.after(self.REFRESH_MS, self._schedule)

    def _redraw(self):
        buf = self._buf
        if buf is None or len(buf) < self.sr * 0.1: return
        max_s = int(self.HISTORY_S * self.sr)
        disp  = buf[-max_s:] if len(buf) > max_s else buf.copy()
        pk    = np.max(np.abs(disp))
        norm  = disp / pk if pk > 0 else disp
        try:
            filt, carrier = _bandpass(norm, self.sr)
            env  = _envelope(filt, self.sr)
            binv = _binary(env, self.sr)
            segs = _segments(binv)
            u    = _unit(segs, self.sr)
            self._draw_wave  (norm, carrier)
            self._draw_fft   (norm, carrier)
            self._draw_binary(binv)
            self._draw_hist  (segs, u)
            self._wf_tick += 1
            if self._wf_tick >= self.WATERFALL_EVERY:
                self._draw_fall(norm, carrier)
                self._wf_tick = 0
            self.canvas.draw_idle()
        except Exception as e:
            print(f"[LivePlotter] _redraw error: {e}")

    def _draw_wave(self, norm, carrier):
        ax = self.ax_wave; ax.cla()
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values(): sp.set_color(C_GREY)
        t = np.linspace(0, len(norm)/self.sr, len(norm))
        ax.plot(t, norm, color=C_GREEN, linewidth=0.4, alpha=0.9)
        ax.set_title("① Oscilloscope  (Raw Waveform)",
                     color=C_WHITE, fontsize=8, pad=3, loc="left")
        ax.set_xlabel("Time (s)", color=C_LABEL, fontsize=7)
        ax.set_ylabel("Amp", color=C_LABEL, fontsize=7)
        ax.set_xlim(0, t[-1]); ax.set_ylim(-1.15, 1.15)
        ax.tick_params(colors=C_LABEL, labelsize=7)
        ax.grid(True, color=C_GREY, linewidth=0.3, alpha=0.4)

    def _draw_fft(self, norm, carrier):
        ax = self.ax_fft; ax.cla()
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values(): sp.set_color(C_GREY)
        win   = np.hanning(len(norm))
        mags  = np.abs(np.fft.rfft(norm * win))
        freqs = np.fft.rfftfreq(len(norm), d=1.0/self.sr)
        mask  = (freqs >= self.FREQ_MIN) & (freqs <= self.FREQ_MAX)
        f, m  = freqs[mask], mags[mask]
        ax.fill_between(f, m, color=C_PEACH, alpha=0.40, linewidth=0)
        ax.plot(f, m, color=C_PEACH, linewidth=0.9)
        if len(m):
            ax.axvline(carrier, color=C_YELLOW, linewidth=1.3, linestyle="--")
            ax.text(carrier+15, m.max()*0.82,
                    f"{carrier:.0f} Hz", color=C_YELLOW, fontsize=7.5, fontweight="bold")
        ax.set_title("② FFT Spectrum  (Frequency Domain)",
                     color=C_WHITE, fontsize=8, pad=3, loc="left")
        ax.set_xlabel("Frequency (Hz)", color=C_LABEL, fontsize=7)
        ax.set_xlim(self.FREQ_MIN, self.FREQ_MAX)
        ax.tick_params(colors=C_LABEL, labelsize=7)
        ax.grid(True, color=C_GREY, linewidth=0.3, alpha=0.4)

    def _draw_fall(self, norm, carrier):
        ax = self.ax_fall; ax.cla()
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values(): sp.set_color(C_GREY)
        nperseg  = min(256, max(32, len(norm)//60))
        noverlap = nperseg * 3 // 4
        f, t, Sxx = scipy_spectrogram(norm, fs=self.sr,
                                      nperseg=nperseg, noverlap=noverlap,
                                      window="hann")
        Sdb   = 10*np.log10(Sxx + 1e-12)
        fmask = (f >= self.FREQ_MIN) & (f <= self.FREQ_MAX)
        Sdb_f = Sdb[fmask]
        if Sdb_f.size == 0 or len(t) < 2:
            return
        vmin = float(np.percentile(Sdb_f, 8))
        vmax = float(np.percentile(Sdb_f, 99))
        if vmin >= vmax:
            vmax = vmin + 1.0
        # freq on x-axis, time on y-axis
        ax.pcolormesh(f[fmask], t, Sdb_f.T, shading="gouraud", cmap="inferno",
                      vmin=vmin, vmax=vmax)
        ax.axvline(carrier, color=C_YELLOW, linewidth=1.1, linestyle="--", alpha=0.85)
        ax.text(carrier + 15, t[-1] * 0.95,
                f"{carrier:.0f} Hz", color=C_YELLOW, fontsize=7.5,
                ha="left", fontweight="bold")
        ax.set_title("③ Waterfall / Spectrogram  —  Morse appears as bright vertical stripe",
                     color=C_WHITE, fontsize=8, pad=3, loc="left")
        ax.set_xlabel("Freq (Hz)", color=C_LABEL, fontsize=7)
        ax.set_ylabel("Time (s)", color=C_LABEL, fontsize=7)
        ax.set_xlim(self.FREQ_MIN, self.FREQ_MAX)
        ax.tick_params(colors=C_LABEL, labelsize=7)

    def _draw_binary(self, binv):
        ax = self.ax_bin; ax.cla()
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values(): sp.set_color(C_GREY)
        t = np.linspace(0, len(binv)/self.sr, len(binv))
        ax.fill_between(t, binv.astype(float),
                        step="pre", color=C_TEAL, alpha=0.65, linewidth=0)
        ax.step(t, binv.astype(float), color=C_TEAL, linewidth=0.7, where="pre")
        ax.set_title("④ Binary Signal  (ON = Dit or Dah)",
                     color=C_WHITE, fontsize=8, pad=3, loc="left")
        ax.set_xlabel("Time (s)", color=C_LABEL, fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["OFF", "ON"], color=C_LABEL, fontsize=7)
        ax.set_xlim(0, t[-1]); ax.set_ylim(-0.15, 1.25)
        ax.tick_params(colors=C_LABEL, labelsize=7)
        ax.grid(True, axis="x", color=C_GREY, linewidth=0.3, alpha=0.4)

    def _draw_hist(self, segs, u):
        ax = self.ax_hist; ax.cla()
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values(): sp.set_color(C_GREY)
        ms    = 1000.0 / self.sr
        dits  = [d*ms for s,d in segs if s==1 and d < u*2]
        dahs  = [d*ms for s,d in segs if s==1 and d >= u*2]
        sps   = [d*ms for s,d in segs if s==0 and d >= u*0.5]
        all_v = dits + dahs + sps
        if not all_v:
            ax.text(0.5, 0.5, "No pulses yet",
                    transform=ax.transAxes, ha="center", va="center",
                    color=C_GREY, fontsize=8, style="italic")
            ax.set_title("⑤ Dit / Dah / Space Durations",
                         color=C_WHITE, fontsize=8, pad=3, loc="left"); return
        top  = min(max(all_v), u*14*ms)
        bins = np.linspace(0, top, 32)
        if dits: ax.hist(dits, bins=bins, color=C_GREEN, alpha=0.85,
                         label=f"Dits ({len(dits)})", edgecolor="none")
        if dahs: ax.hist(dahs, bins=bins, color=C_RED,   alpha=0.85,
                         label=f"Dahs ({len(dahs)})", edgecolor="none")
        if sps:  ax.hist(sps,  bins=bins, color=C_BLUE,  alpha=0.55,
                         label=f"Spaces ({len(sps)})", edgecolor="none")
        split = u*2*ms
        ax.axvline(split, color=C_YELLOW, linewidth=1.2, linestyle="--")
        ax.text(split*1.04, ax.get_ylim()[1]*0.88,
                f"split\n{split:.0f}ms", color=C_YELLOW, fontsize=6.5)
        ax.set_title("⑤ Dit / Dah / Space Durations",
                     color=C_WHITE, fontsize=8, pad=3, loc="left")
        ax.set_xlabel("Duration (ms)", color=C_LABEL, fontsize=7)
        ax.set_ylabel("Count",         color=C_LABEL, fontsize=7)
        ax.tick_params(colors=C_LABEL, labelsize=7)
        ax.grid(True, axis="y", color=C_GREY, linewidth=0.3, alpha=0.4)
        ax.legend(fontsize=7, facecolor=SURFACE0, edgecolor=C_GREY,
                  labelcolor=C_WHITE, loc="upper right")


# ══════════════════════════════════════════════════════════════════════════════
# LetterTilePanel — scrolling row of decoded letter cards
# ══════════════════════════════════════════════════════════════════════════════

class LetterTilePanel(tk.Frame):
    """
    Horizontal scrolling row of letter 'tiles'.
    Each tile shows:  large letter on top, Morse code dots/dashes below.
    New tiles are added from the right; old ones scroll off the left.
    """
    TILE_W    = 58
    TILE_H    = 72
    MAX_TILES = 22     # how many tiles to keep in view

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG3, **kwargs)
        self._tiles   = []      # list of (letter, frame widget)
        self._prev    = ""      # previous full decoded string

        # scrollable canvas inside this frame
        self._canvas  = tk.Canvas(self, bg=BG3, height=self.TILE_H + 8,
                                  highlightthickness=0, bd=0)
        self._canvas.pack(fill=tk.X, expand=True)
        self._inner   = tk.Frame(self._canvas, bg=BG3)
        self._window  = self._canvas.create_window(
            0, 4, anchor="nw", window=self._inner)

    def update_text(self, text: str):
        """Called every time the decoder produces a new string."""
        if text == self._prev:
            return
        self._prev = text

        # Rebuild tiles for each character in the decoded text
        for w in self._inner.winfo_children():
            w.destroy()
        self._tiles = []

        chars = [c for c in text if c.strip()]  # skip spaces for tiles
        # Only show the last MAX_TILES characters
        chars = chars[-self.MAX_TILES:]

        for i, ch in enumerate(chars):
            colour = _TILE_COLOURS[ord(ch) % len(_TILE_COLOURS)]
            morse  = _MORSE.get(ch.upper(), "?")
            self._make_tile(ch.upper(), morse, colour)

        self._canvas.update_idletasks()
        self._canvas.config(scrollregion=self._canvas.bbox("all"))
        # scroll to the right to show latest tile
        self._canvas.xview_moveto(1.0)

    def _make_tile(self, letter, morse, colour):
        tile = tk.Frame(self._inner, bg=SURFACE0,
                        width=self.TILE_W, height=self.TILE_H,
                        padx=3, pady=3)
        tile.pack(side=tk.LEFT, padx=3, pady=4)
        tile.pack_propagate(False)

        # coloured top accent bar
        tk.Frame(tile, bg=colour, height=3).pack(fill=tk.X)

        # big letter
        tk.Label(tile, text=letter, bg=SURFACE0, fg=colour,
                 font=("Segoe UI", 22, "bold")).pack(pady=(2, 0))

        # morse dots/dashes
        tk.Label(tile, text=morse, bg=SURFACE0, fg=C_SUBTEXT,
                 font=("Courier New", 9)).pack()


# ══════════════════════════════════════════════════════════════════════════════
# DecoderUI — main application window
# ══════════════════════════════════════════════════════════════════════════════

class DecoderUI:

    def __init__(self, initial_file=None):
        self._q           = queue.Queue()
        self._audio_input = None
        self._decoder     = None
        self._plotter     = None
        self._file        = initial_file

        self._build_root()
        self._build_titlebar()
        self._build_body()
        self._build_statusbar()

        self._root.after(50, self._poll_queue)
        if initial_file:
            self._root.after(300, lambda: self._load_file(initial_file))

    # ── root window ───────────────────────────────────────────────────────────

    def _build_root(self):
        self._root = tk.Tk()
        self._root.title("Morse Code Decoder  ·  Live Dashboard")
        self._root.geometry("1480x860")
        self._root.minsize(1100, 700)
        self._root.configure(bg=BG)
        self._root.resizable(True, True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── title bar ─────────────────────────────────────────────────────────────

    def _build_titlebar(self):
        bar = tk.Frame(self._root, bg=BG2, pady=0)
        bar.pack(fill=tk.X)

        # left: app name + dot indicator
        left = tk.Frame(bar, bg=BG2)
        left.pack(side=tk.LEFT, padx=12, pady=8)

        self._dot = tk.Label(left, text="●", bg=BG2, fg=SURFACE1,
                             font=("Segoe UI", 11))
        self._dot.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(left, text="Morse Code Decoder", bg=BG2, fg=C_WHITE,
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        tk.Label(left, text="  Live Dashboard", bg=BG2, fg=OVERLAY,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT)

        # right: buttons
        right = tk.Frame(bar, bg=BG2)
        right.pack(side=tk.RIGHT, padx=10, pady=6)

        self._make_btn(right, "📂  Open WAV",     self._browse,       C_BLUE  ).pack(side=tk.LEFT, padx=3)
        self._make_btn(right, "🎙  Microphone",   self._start_mic,    C_MAUVE ).pack(side=tk.LEFT, padx=3)
        self._make_btn(right, "🔊  System Audio", self._start_system, C_GREEN ).pack(side=tk.LEFT, padx=3)
        self._make_btn(right, "⏹  Stop",          self._stop,         C_PEACH ).pack(side=tk.LEFT, padx=3)
        self._make_btn(right, "🗑  Clear",         self._clear,        OVERLAY ).pack(side=tk.LEFT, padx=3)

        # source label pill
        self._source_lbl = tk.Label(right, text="  No source  ",
                                    bg=SURFACE0, fg=C_SUBTEXT,
                                    font=("Segoe UI", 8),
                                    padx=8, pady=2, relief=tk.FLAT)
        self._source_lbl.pack(side=tk.LEFT, padx=(12, 0))

    def _make_btn(self, parent, text, cmd, colour):
        btn = tk.Label(parent, text=text, bg=SURFACE0, fg=colour,
                       font=("Segoe UI", 9, "bold"),
                       padx=10, pady=5, cursor="hand2", relief=tk.FLAT)
        btn.bind("<Button-1>", lambda e: cmd())
        btn.bind("<Enter>",    lambda e: btn.config(bg=SURFACE1))
        btn.bind("<Leave>",    lambda e: btn.config(bg=SURFACE0))
        return btn

    # ── body: left plots + right panel ────────────────────────────────────────

    def _build_body(self):
        body = tk.Frame(self._root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── LEFT: 5-panel signal visualiser ───────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._plotter = LivePlotter(left, sr=8000)

        # ── RIGHT: output panel ────────────────────────────────────────────────
        right = tk.Frame(body, bg=BG2, width=440)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(6, 0))
        right.pack_propagate(False)

        # ── section header helper ──────────────────────────────────────────────
        def section(parent, title):
            tk.Label(parent, text=title, bg=BG2, fg=C_BLUE,
                     font=("Segoe UI", 8, "bold"),
                     padx=10, pady=4).pack(anchor=tk.W)
            tk.Frame(parent, bg=SURFACE0, height=1).pack(fill=tk.X, padx=10)

        # ── 1. Letter tiles ────────────────────────────────────────────────────
        section(right, "DECODED LETTERS")
        self._tiles = LetterTilePanel(right, height=88)
        self._tiles.pack(fill=tk.X, padx=6, pady=(4, 8))

        # ── 2. Morse symbols ───────────────────────────────────────────────────
        section(right, "MORSE SYMBOLS")
        self._morse_area = scrolledtext.ScrolledText(
            right, font=("Courier New", 10),
            bg=BG3, fg=C_YELLOW,
            insertbackground=C_WHITE, selectbackground=C_BLUE,
            wrap=tk.WORD, relief=tk.FLAT, padx=8, pady=6,
            height=4, state=tk.DISABLED, borderwidth=0,
        )
        self._morse_area.pack(fill=tk.X, padx=6, pady=(4, 8))

        # ── 3. Full decoded text ───────────────────────────────────────────────
        section(right, "FULL DECODED TEXT")
        self._text_area = scrolledtext.ScrolledText(
            right, font=("Courier New", 13),
            bg=BG3, fg=C_WHITE,
            insertbackground=C_WHITE, selectbackground=C_BLUE,
            wrap=tk.WORD, relief=tk.FLAT, padx=10, pady=8,
            state=tk.DISABLED, borderwidth=0,
        )
        self._text_area.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 0))

    # ── status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = tk.Frame(self._root, bg=BG2, height=28)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        inner = tk.Frame(bar, bg=BG2)
        inner.pack(fill=tk.X, padx=10, pady=4)

        # SNR bar
        tk.Label(inner, text="SNR", bg=BG2, fg=OVERLAY,
                 font=("Segoe UI", 7)).pack(side=tk.LEFT)
        self._snr_canvas = tk.Canvas(inner, bg=SURFACE0, width=80, height=10,
                                     highlightthickness=0, bd=0)
        self._snr_canvas.pack(side=tk.LEFT, padx=(4, 12))
        self._snr_bar = self._snr_canvas.create_rectangle(
            0, 0, 0, 10, fill=C_GREEN, outline="")

        # status pills
        self._pill_freq = self._make_pill(inner, "--- Hz")
        self._pill_wpm  = self._make_pill(inner, "-- WPM")
        self._pill_snr  = self._make_pill(inner, "WAITING", OVERLAY)

        # right: main status text
        self._status_var = tk.StringVar(value="Ready — open a file or start audio input")
        tk.Label(inner, textvariable=self._status_var,
                 bg=BG2, fg=OVERLAY, font=("Segoe UI", 8),
                 anchor=tk.E).pack(side=tk.RIGHT, padx=4)

    def _make_pill(self, parent, text, colour=C_TEAL):
        lbl = tk.Label(parent, text=text, bg=SURFACE0, fg=colour,
                       font=("Segoe UI", 8, "bold"), padx=7, pady=1)
        lbl.pack(side=tk.LEFT, padx=3)
        return lbl

    # ── actions ───────────────────────────────────────────────────────────────

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select a WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if p: self._load_file(p)

    def _load_file(self, path):
        if not os.path.exists(path):
            messagebox.showerror("File not found", f"Cannot find:\n{path}"); return
        self._stop(); self._clear()
        self._file = path
        self._source_lbl.config(text=f"  {os.path.basename(path)}  ")
        self._set_status(f"Loading {os.path.basename(path)}…")
        self._start_pipeline(file_path=path)

    def _start_mic(self):
        self._stop(); self._clear()
        self._file = None
        self._source_lbl.config(text="  🎙 Microphone  ")
        try:
            self._start_pipeline(mic=True)
            self._set_status("🎙 Listening on microphone…")
        except RuntimeError as e:
            messagebox.showwarning("Mic unavailable", str(e))

    def _start_system(self):
        self._stop(); self._clear()
        self._file = None
        try:
            from src.audio_input import _PAW_AVAILABLE
            if not _PAW_AVAILABLE:
                self._show_install_pawpatch(); return
            self._source_lbl.config(text="  🔊 WASAPI Loopback  ")
            self._set_status("🔊 Opening WASAPI loopback…")
            self._start_pipeline(system_audio=True)
            self._set_status("🔊 Ready — play your YouTube video now!")
        except Exception as e:
            self._set_status(f"System audio error: {e}")
            messagebox.showerror("System Audio Error", str(e))

    def _start_pipeline(self, file_path=None, mic=False, system_audio=False):
        from src.audio_input import AudioInput
        ai  = AudioInput(sample_rate=8000)
        dec = StreamingDecoder(
            sr       = 8000,
            on_text  = lambda t: self._q.put(("text",   t)),
            on_status= lambda s: self._q.put(("status", s)),
            on_signal= lambda b: self._q.put(("signal", b)),
        )
        ai.register_callback(dec.push)
        if system_audio:
            ai._on_error = lambda m: self._q.put(("stream_error", m))
            ai.start_system_audio()
        elif mic:
            ai.start_microphone()
        else:
            ai.stream_file_async(file_path, realtime=True)
        self._audio_input = ai
        self._decoder     = dec
        self._plotter.start(self._root)
        self._dot.config(fg=C_GREEN)    # green dot = active

    def _stop(self):
        self._plotter.stop(); self._plotter.reset()
        if self._audio_input: self._audio_input.stop(); self._audio_input = None
        if self._decoder:     self._decoder.reset();    self._decoder     = None
        self._dot.config(fg=SURFACE1)
        self._set_status("Stopped.")
        self._update_status_bar(0, 0, 0)

    def _clear(self):
        self._tiles.update_text("")
        for area in (self._morse_area, self._text_area):
            area.config(state=tk.NORMAL)
            area.delete("1.0", tk.END)
            area.config(state=tk.DISABLED)

    # ── queue polling ─────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if   kind == "text":         self._show_text(payload)
                elif kind == "status":       self._handle_status(payload)
                elif kind == "signal":       self._plotter.push(payload)
                elif kind == "stream_error": self._set_status("⚠ Stream failed"); messagebox.showerror("Audio Error", payload)
        except queue.Empty:
            pass
        self._root.after(50, self._poll_queue)

    def _handle_status(self, payload):
        kind, snr, freq, wpm = payload
        if kind == "buffering":
            self._set_status("Buffering audio…")
            self._update_status_bar(0, 0, 0)
        elif kind == "live":
            self._update_status_bar(snr, freq, wpm)
        elif kind == "error":
            self._set_status("Decode error")

    def _update_status_bar(self, snr, freq, wpm):
        # SNR meter bar (0–50 dB range)
        pct = min(1.0, max(0.0, (snr - 10) / 40.0))
        bar_w = int(80 * pct)
        colour = C_GREEN if snr >= 28 else C_YELLOW if snr >= 20 else C_RED
        self._snr_canvas.coords(self._snr_bar, 0, 0, bar_w, 10)
        self._snr_canvas.itemconfig(self._snr_bar, fill=colour)

        # pills
        self._pill_freq.config(text=f"{freq:.0f} Hz" if freq else "--- Hz")
        self._pill_wpm .config(text=f"{wpm} WPM"     if wpm  else "-- WPM")
        if   snr >= 28: self._pill_snr.config(text="GOOD",    fg=C_GREEN)
        elif snr >= 20: self._pill_snr.config(text="OK",      fg=C_YELLOW)
        elif snr >  0:  self._pill_snr.config(text="WEAK",    fg=C_RED)
        else:           self._pill_snr.config(text="WAITING", fg=OVERLAY)

    # ── text display ──────────────────────────────────────────────────────────

    def _show_text(self, decoded: str):
        # letter tiles
        self._tiles.update_text(decoded)

        # full text area
        self._text_area.config(state=tk.NORMAL)
        self._text_area.delete("1.0", tk.END)
        self._text_area.insert(tk.END, decoded)
        self._text_area.see(tk.END)
        self._text_area.config(state=tk.DISABLED)

        # morse symbols
        try:
            from src.constants import TEXT_TO_MORSE
            syms = "  ".join(
                TEXT_TO_MORSE.get(c, "?") if c != " " else "/"
                for c in decoded
            )
        except Exception:
            syms = decoded
        self._morse_area.config(state=tk.NORMAL)
        self._morse_area.delete("1.0", tk.END)
        self._morse_area.insert(tk.END, syms)
        self._morse_area.see(tk.END)
        self._morse_area.config(state=tk.DISABLED)

    def _set_status(self, msg):
        self._status_var.set(msg)

    # ── popups ────────────────────────────────────────────────────────────────

    def _show_install_pawpatch(self):
        win = tk.Toplevel(self._root)
        win.title("Install required"); win.configure(bg=BG)
        win.geometry("520x280"); win.resizable(False, False)
        tk.Label(win, text="🔊  One Package Needed",
                 bg=BG, fg=C_YELLOW,
                 font=("Segoe UI", 12, "bold")).pack(pady=(18, 6))
        msg = ("Run this in your terminal, then restart:\n\n"
               "    pip install pyaudiowpatch\n\n"
               "Then click  🔊 System Audio  and play your YouTube video.\n"
               "No Stereo Mix or VB-Cable needed.")
        tk.Label(win, text=msg, bg=BG, fg=C_WHITE,
                 font=("Courier New", 10),
                 justify=tk.LEFT, padx=24).pack(fill=tk.BOTH, expand=True)
        tk.Button(win, text="OK", command=win.destroy,
                  bg=C_BLUE, fg=BG, font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, padx=20, pady=6,
                  cursor="hand2").pack(pady=(0, 16))

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop(); self._root.destroy()

    def run(self):
        self._root.mainloop()


if __name__ == "__main__":
    DecoderUI(sys.argv[1] if len(sys.argv) > 1 else None).run()
