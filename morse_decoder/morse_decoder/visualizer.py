"""
visualizer.py — All-in-One Morse Signal Dashboard
==================================================
Five panels on one screen (assignment: "Signal Visualization"):

  Panel 1  (top-left)     Oscilloscope — raw time-domain waveform
  Panel 2  (top-right)    FFT Spectrum — frequency-domain, carrier marked
  Panel 3  (middle full)  Spectrogram / Waterfall — time × frequency heatmap
  Panel 4  (lower-left)   Binary Signal — smoothed ON/OFF envelope
  Panel 5  (lower-right)  Dit / Dah / Space histogram

What each panel demonstrates to the examiner
---------------------------------------------
- Oscilloscope  : raw noisy audio — dots/dashes appear as burst groups.
- FFT Spectrum  : bandpass filter is centred on the correct carrier tone;
                  the yellow marker shows the auto-detected frequency.
- Waterfall     : SDR-style view — Morse pattern is a bright horizontal
                  stripe; dot/dash durations are visible directly.
- Binary Signal : clean rectangular pulses after envelope + threshold.
- Histogram     : dit (green) vs dah (red) vs space (blue) cluster
                  separation proves the adaptive unit-time algorithm.

Usage
-----
    python visualizer.py                       # uses noisy_paragraph.wav
    python visualizer.py my_audio.wav
    python visualizer.py my_audio.wav out.png  # save to PNG
"""

from __future__ import annotations

import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.signal import welch

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — dark oscilloscope / SDR theme
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#0D0D1A"
PANEL_BG  = "#11111F"
GREEN     = "#00FF88"
ORANGE    = "#FF8800"
YELLOW    = "#FFE500"
CYAN      = "#00CFFF"
RED       = "#FF4466"
BLUE      = "#4488FF"
GREY      = "#555577"
LABEL_COL = "#AAAACC"
WHITE     = "#E8E8FF"


class Visualizer:
    """
    All-in-one Morse signal dashboard.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz (typically 8 000).
    data : np.ndarray
        Raw PCM samples (int16 or float32, mono or stereo).
    title : str
        Label shown in the window title.
    max_seconds : float
        Clip audio to this many seconds for display (default 10).
    freq_min, freq_max : int
        Frequency display range in Hz.
    """

    def __init__(
        self,
        sample_rate: int,
        data:        np.ndarray,
        title:       str   = "Morse Signal",
        max_seconds: float = 10.0,
        freq_min:    int   = 200,
        freq_max:    int   = 1500,
    ) -> None:
        if data.ndim > 1:
            data = data[:, 0]

        self.sr       = sample_rate
        self.title    = title
        self.freq_min = freq_min
        self.freq_max = freq_max

        # Normalise to float32 [-1, 1]
        raw  = data.astype(np.float32)
        peak = np.max(np.abs(raw))
        self.raw_norm = raw / peak if peak > 0 else raw

        # Clip for display
        max_samp      = int(max_seconds * sample_rate)
        self.display  = self.raw_norm[:max_samp]

        # Pre-compute derived signals
        self._filtered  = self._bandpass(self.display)
        self._envelope  = self._smooth_envelope(self._filtered)
        self._binary    = self._threshold(self._envelope)
        self._segments  = self._get_segments(self._binary)
        self._unit      = self._estimate_unit(self._segments)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def show(self, save_path: str | None = None) -> None:
        """Render the dashboard.  Pass save_path to write a PNG."""
        fig = self._build_figure()
        if save_path:
            fig.savefig(save_path, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[Visualizer] Saved → {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def get_peak_frequency(self) -> float:
        """Return the dominant carrier frequency in Hz."""
        freqs, mags = self._fft_spectrum(self.display)
        mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        return float(freqs[mask][np.argmax(mags[mask])]) if np.any(mask) else 700.0

    # ──────────────────────────────────────────────────────────────────────────
    # Signal processing
    # ──────────────────────────────────────────────────────────────────────────

    def _bandpass(self, data: np.ndarray) -> np.ndarray:
        f, psd = welch(data, self.sr, nperseg=min(1024, len(data)))
        mask    = (f > 200) & (f < 1400)
        carrier = float(f[mask][np.argmax(psd[mask])]) if np.any(mask) else 700.0
        lo, hi  = max(50, carrier - 150), min(self.sr / 2 - 1, carrier + 150)
        nyq     = self.sr / 2
        b, a    = butter(4, [lo / nyq, hi / nyq], btype="band")
        return lfilter(b, a, data)

    def _smooth_envelope(self, data: np.ndarray) -> np.ndarray:
        win = max(1, int(self.sr * 0.006))
        return np.convolve(np.abs(data), np.ones(win) / win, mode="same")

    def _threshold(self, env: np.ndarray) -> np.ndarray:
        lo     = np.percentile(env, 5)
        hi     = np.percentile(env, 95)
        thresh = lo + (hi - lo) * 0.45
        return (env > thresh).astype(np.int8)

    def _get_segments(self, binary: np.ndarray) -> list[tuple[int, int]]:
        changes = np.diff(binary.astype(int))
        idx     = [0] + list(np.where(changes != 0)[0] + 1) + [len(binary)]
        return [(int(binary[idx[i]]), idx[i + 1] - idx[i])
                for i in range(len(idx) - 1)]

    def _estimate_unit(self, segments: list[tuple[int, int]]) -> float:
        on_durs = [d for s, d in segments if s == 1 and d > 0]
        if not on_durs:
            return float(self.sr * 0.08)
        med  = float(np.median(on_durs))
        dits = [d for d in on_durs if d < med]
        return float(np.mean(dits)) if dits else med / 2.0

    def _fft_spectrum(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        win  = np.hanning(len(data))
        mags = np.abs(np.fft.rfft(data * win))
        freq = np.fft.rfftfreq(len(data), d=1.0 / self.sr)
        return freq, mags

    # ──────────────────────────────────────────────────────────────────────────
    # Figure assembly
    # ──────────────────────────────────────────────────────────────────────────

    def _build_figure(self) -> plt.Figure:
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(16, 11), facecolor=BG)
        fig.suptitle(
            f"Morse Code Decoder  —  {self.title}",
            color=WHITE, fontsize=13, fontweight="bold", y=0.985,
        )

        gs = gridspec.GridSpec(
            3, 2, figure=fig,
            height_ratios=[1, 1.1, 1],
            hspace=0.52, wspace=0.32,
            top=0.955, bottom=0.055,
            left=0.065, right=0.97,
        )

        ax_wave  = fig.add_subplot(gs[0, 0])
        ax_fft   = fig.add_subplot(gs[0, 1])
        ax_wfall = fig.add_subplot(gs[1, :])   # full-width
        ax_bin   = fig.add_subplot(gs[2, 0])
        ax_hist  = fig.add_subplot(gs[2, 1])

        for ax in (ax_wave, ax_fft, ax_wfall, ax_bin, ax_hist):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=LABEL_COL, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(GREY)

        self._panel_waveform   (ax_wave)
        self._panel_fft        (ax_fft)
        self._panel_waterfall  (ax_wfall)
        self._panel_binary     (ax_bin)
        self._panel_ditdah_hist(ax_hist)

        return fig

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 1 — Oscilloscope
    # ──────────────────────────────────────────────────────────────────────────

    def _panel_waveform(self, ax: plt.Axes) -> None:
        t = np.linspace(0, len(self.display) / self.sr, len(self.display))
        ax.plot(t, self.display, color=GREEN, linewidth=0.35, alpha=0.9)
        ax.set_title("① Oscilloscope  (Raw Waveform)", color=WHITE,
                     fontsize=9, pad=4, loc="left")
        ax.set_xlabel("Time (s)",  color=LABEL_COL, fontsize=8)
        ax.set_ylabel("Amplitude", color=LABEL_COL, fontsize=8)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-1.15, 1.15)
        ax.grid(True, color=GREY, linewidth=0.4, alpha=0.5)
        ax.axhline(0, color=GREY, linewidth=0.5)
        ax.text(0.99, 0.93, "Dots & dashes = pulse bursts",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=LABEL_COL, style="italic")

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 2 — FFT Spectrum
    # ──────────────────────────────────────────────────────────────────────────

    def _panel_fft(self, ax: plt.Axes) -> None:
        freqs, mags = self._fft_spectrum(self.display)
        mask        = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        f, m        = freqs[mask], mags[mask]

        ax.fill_between(f, m, color=ORANGE, alpha=0.45, linewidth=0)
        ax.plot(f, m, color=ORANGE, linewidth=0.9)

        if len(m) > 0:
            pk_f = f[np.argmax(m)]
            pk_m = m[np.argmax(m)]
            ax.axvline(pk_f, color=YELLOW, linewidth=1.3,
                       linestyle="--", alpha=0.9)
            ax.text(
                pk_f + (self.freq_max - self.freq_min) * 0.02,
                pk_m * 0.85,
                f"Carrier\n{pk_f:.0f} Hz",
                color=YELLOW, fontsize=7.5, va="top",
            )

        ax.set_title("② FFT Spectrum  (Frequency Domain)", color=WHITE,
                     fontsize=9, pad=4, loc="left")
        ax.set_xlabel("Frequency (Hz)", color=LABEL_COL, fontsize=8)
        ax.set_ylabel("Magnitude",      color=LABEL_COL, fontsize=8)
        ax.set_xlim(self.freq_min, self.freq_max)
        ax.grid(True, color=GREY, linewidth=0.4, alpha=0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 3 — Spectrogram / Waterfall
    # ──────────────────────────────────────────────────────────────────────────

    def _panel_waterfall(self, ax: plt.Axes) -> None:
        nperseg  = min(256, max(32, len(self.display) // 60))
        noverlap = nperseg * 3 // 4

        f, t, Sxx = scipy_spectrogram(
            self.display, fs=self.sr,
            nperseg=nperseg, noverlap=noverlap, window="hann",
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        fmask  = (f >= self.freq_min) & (f <= self.freq_max)
        f_show = f[fmask]
        S_show = Sxx_db[fmask, :]

        vmin = float(np.percentile(S_show, 8))
        vmax = float(np.percentile(S_show, 99))

        img = ax.pcolormesh(t, f_show, S_show,
                            shading="gouraud", cmap="inferno",
                            vmin=vmin, vmax=vmax)
        cb = plt.colorbar(img, ax=ax, pad=0.008, aspect=28)
        cb.set_label("Power (dB)", color=LABEL_COL, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=LABEL_COL, labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=LABEL_COL)

        peak = self.get_peak_frequency()
        ax.axhline(peak, color=YELLOW, linewidth=1.1, linestyle="--", alpha=0.8)
        ax.text(t[-1] * 0.995, peak + 18,
                f"{peak:.0f} Hz", color=YELLOW, fontsize=7.5, ha="right")

        ax.set_title(
            "③ Waterfall / Spectrogram  —  Morse appears as bright horizontal stripe",
            color=WHITE, fontsize=9, pad=4, loc="left",
        )
        ax.set_xlabel("Time (s)",       color=LABEL_COL, fontsize=8)
        ax.set_ylabel("Frequency (Hz)", color=LABEL_COL, fontsize=8)

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 4 — Binary Signal
    # ──────────────────────────────────────────────────────────────────────────

    def _panel_binary(self, ax: plt.Axes) -> None:
        t = np.linspace(0, len(self.display) / self.sr, len(self._binary))
        ax.fill_between(t, self._binary.astype(float),
                        step="pre", color=CYAN, alpha=0.75, linewidth=0)
        ax.step(t, self._binary.astype(float), color=CYAN,
                linewidth=0.7, where="pre")

        ax.set_title("④ Binary Signal  (ON = Dit or Dah)", color=WHITE,
                     fontsize=9, pad=4, loc="left")
        ax.set_xlabel("Time (s)", color=LABEL_COL, fontsize=8)
        ax.set_ylabel("State",    color=LABEL_COL, fontsize=8)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-0.15, 1.25)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["OFF", "ON"], color=LABEL_COL, fontsize=8)
        ax.grid(True, axis="x", color=GREY, linewidth=0.4, alpha=0.5)
        ax.text(0.99, 0.93, "Each ON block = one Dit or Dah",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=LABEL_COL, style="italic")

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 5 — Dit / Dah / Space Duration Histogram
    # ──────────────────────────────────────────────────────────────────────────

    def _panel_ditdah_hist(self, ax: plt.Axes) -> None:
        unit = max(1.0, self._unit)
        ms   = 1000.0 / self.sr  # samples → ms

        dits   = [d * ms for s, d in self._segments if s == 1 and d < unit * 2]
        dahs   = [d * ms for s, d in self._segments if s == 1 and d >= unit * 2]
        spaces = [d * ms for s, d in self._segments if s == 0 and d >= unit * 0.5]

        all_vals = dits + dahs + spaces
        if not all_vals:
            ax.text(0.5, 0.5, "No segments detected",
                    ha="center", va="center",
                    color=LABEL_COL, transform=ax.transAxes)
            return

        max_ms = max(all_vals)
        bins   = np.linspace(0, min(max_ms, unit * 12 * ms), 36)

        if dits:
            ax.hist(dits,   bins=bins, color=GREEN, alpha=0.80,
                    label=f"Dits ({len(dits)})",    edgecolor="none")
        if dahs:
            ax.hist(dahs,   bins=bins, color=RED,   alpha=0.80,
                    label=f"Dahs ({len(dahs)})",    edgecolor="none")
        if spaces:
            ax.hist(spaces, bins=bins, color=BLUE,  alpha=0.55,
                    label=f"Spaces ({len(spaces)})", edgecolor="none")

        split_ms = unit * 2 * ms
        ax.axvline(split_ms, color=YELLOW, linewidth=1.2, linestyle="--", alpha=0.9)
        ax.text(split_ms + max_ms * 0.01, ax.get_ylim()[1] * 0.92,
                f"Dit/Dah\n split\n{split_ms:.0f} ms",
                color=YELLOW, fontsize=6.5)

        ax.set_title("⑤ Dit / Dah / Space Durations", color=WHITE,
                     fontsize=9, pad=4, loc="left")
        ax.set_xlabel("Duration (ms)", color=LABEL_COL, fontsize=8)
        ax.set_ylabel("Count",         color=LABEL_COL, fontsize=8)
        ax.grid(True, axis="y", color=GREY, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=7.5, facecolor="#1A1A2E",
                  edgecolor=GREY, labelcolor=WHITE, loc="upper right")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry-point
# ─────────────────────────────────────────────────────────────────────────────

def visualize_wav(file_path: str, save_path: str | None = None) -> None:
    """Load a WAV file and show the 5-panel dashboard."""
    try:
        fs, data = wavfile.read(file_path)
    except Exception as exc:
        print(f"[Visualizer] Cannot read '{file_path}': {exc}")
        return

    viz = Visualizer(fs, data, title=file_path)
    print(f"[Visualizer] Carrier frequency : {viz.get_peak_frequency():.1f} Hz")
    print(f"[Visualizer] Estimated dit     : {viz._unit / fs * 1000:.1f} ms")
    viz.show(save_path=save_path)


if __name__ == "__main__":
    wav_  = sys.argv[1] if len(sys.argv) > 1 else "noisy_paragraph.wav"
    out_  = sys.argv[2] if len(sys.argv) > 2 else None
    visualize_wav(wav_, out_)
