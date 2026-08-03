# Morse Code Decoder - Live Dashboard

A real-time Morse code decoder with a live GUI dashboard. Audio is captured from a WAV file, microphone, or system speakers and decoded to text using a DSP signal processing pipeline. Decoded output is refined by a **probabilistic N-gram corrector** that fills gaps and fixes common Morse confusions. Results are displayed in a 5-panel signal visualiser with letter tiles and a live text transcript.

---

## Features

- **Three audio sources**: WAV file playback, microphone (via `sounddevice`), or system audio (WASAPI loopback on Windows via `pyaudiowpatch`)
- **Adaptive noise handling**: SNR-gated decoder rejects pure noise; adaptive threshold recovers weak signals
- **N-gram probabilistic corrector**: a character-level interpolated trigram language model that fills `[?]` gaps and auto-corrects common Morse confusions (E↔I, S↔H, T↔M, etc.) using bidirectional context scoring against an English + amateur-radio corpus
- **Live 5-panel signal visualiser**: oscilloscope, FFT spectrum, waterfall spectrogram, binary signal, and dit/dah duration histogram
- **Letter tile display**: each decoded character rendered as a coloured card with its Morse pattern underneath
- **Live status bar**: real-time SNR meter, carrier frequency pill, and WPM pill

---

## Quick Start

```bash
pip install -r requirements.txt

python main.py                  # open GUI
python main.py myfile.wav       # open GUI and decode a WAV immediately
python main.py --mic            # open GUI and start microphone immediately
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        main.py                           │
│  Parses CLI args (--mic, WAV path), launches DecoderUI   │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                      src/ui.py                           │
│   DecoderUI ─► LivePlotter (5 panels)                    │
│             ─► LetterTilePanel                           │
│             ─► StreamingDecoder (coordinator)            │
└────┬─────────────────────────────────┬───────────────────┘
     │                                 │
     ▼                                 ▼
┌─────────────┐              ┌──────────────────────┐
│src/audio_   │              │     src/engine.py    │
│input.py     │──samples──►  │     MorseEngine      │
│AudioInput   │              │  (DSP decode)        │
│ - WAV file  │              └──────────┬───────────┘
│ - Mic (sd)  │                         │ raw decoded text
│ - WASAPI    │                         ▼
└─────────────┘              ┌──────────────────────┐
                             │   src/corrector.py   │
                             │   MorseCorrector     │
                             │  (N-gram correction) │
                             └──────────┬───────────┘
                                        │ corrected text
                                        ▼
                                 displayed in UI

src/constants.py ─────────────────────► used by engine + ui
```

---

## File Reference

### Entry Point

| File | Role |
|------|------|
| `main.py` | Parses CLI args (`--mic`, WAV path), checks tkinter availability, launches [DecoderUI](file:///d:/Masters/2026-Summer%20Semester/OOP/Morse-Code-Decoder/Morse-Code-Decoder/src/ui.py#478-816). |

### `src/` — Core Library

| File | Role |
|------|------|
| `constants.py` | Defines `MORSE_MAP` (Morse pattern → character) and `TEXT_TO_MORSE` (inverse). Single source of truth for the symbol table — both `engine.py` and [ui.py](file:///d:/Masters/2026-Summer%20Semester/OOP/Morse-Code-Decoder/Morse-Code-Decoder/src/ui.py) import from here. |
| `engine.py` | `MorseEngine`: the primary DSP decoder. Pipeline: auto-detect carrier frequency via Welch PSD → bandpass filter (±200 Hz) → amplitude envelope → adaptive binary threshold → pulse duration segmentation → dit/dah/space classification → `MORSE_MAP` lookup. Includes an SNR gate (28 dB threshold) and a transition-rate check that together reject pure background noise. Also exposes `decode_wav()` as a standalone helper. |
| [corrector.py](file:///d:/Masters/2026-Summer%20Semester/OOP/Morse-Code-Decoder/Morse-Code-Decoder/tests/test_corrector.py) | `MorseCorrector`: probabilistic N-gram post-processor. Uses an interpolated trigram model (λ₁=0.10, λ₂=0.30, λ₃=0.60) trained on an English + amateur-radio corpus. **Pass 1** fills `[?]` gaps via bidirectional context scoring. **Pass 2** fixes common Morse confusion pairs (E↔I, S↔H, T↔M, etc.) when the alternative character is ≥15× more probable. Runs in <1 ms per decode. |
| `audio_input.py` | `AudioInput`: thread-based audio source with a callback/observer interface. Supports WAV file streaming, microphone via `sounddevice`, and system audio via `pyaudiowpatch` WASAPI loopback. Handles resampling to 8 kHz and stereo→mono conversion internally. |
| [ui.py](file:///d:/Masters/2026-Summer%20Semester/OOP/Morse-Code-Decoder/Morse-Code-Decoder/src/ui.py) | The complete GUI. [DecoderUI](file:///d:/Masters/2026-Summer%20Semester/OOP/Morse-Code-Decoder/Morse-Code-Decoder/src/ui.py#478-816) builds the window and toolbar. [StreamingDecoder](file:///d:/Masters/2026-Summer%20Semester/OOP/Morse-Code-Decoder/Morse-Code-Decoder/src/ui.py#121-175) buffers incoming audio chunks, throttles decode calls to every 0.4 s, and routes results to the UI via a thread-safe queue. [LivePlotter](file:///d:/Masters/2026-Summer%20Semester/OOP/Morse-Code-Decoder/Morse-Code-Decoder/src/ui.py#181-401) drives the 5-panel Matplotlib canvas at 150 ms refresh. [LetterTilePanel](file:///d:/Masters/2026-Summer%20Semester/OOP/Morse-Code-Decoder/Morse-Code-Decoder/src/ui.py#407-472) renders decoded characters as coloured cards. Uses the Catppuccin Mocha colour palette. |
| `__init__.py` | Makes `src/` a Python package. |

### Tests

| File | What it tests |
|------|---------------|
| `tests/test_constants.py` | `MORSE_MAP` completeness (26 letters, 10 digits, special characters) and `TEXT_TO_MORSE` invertibility |
| `tests/test_engine.py` | `MorseEngine` frequency detection, SNR estimation, and decoding on clean/noisy/silent signals; `decode_wav()` helper |
| `tests/test_audio_input.py` | `AudioInput` initialisation, callback dispatch, resampling, and WAV file streaming |
| [tests/test_corrector.py](file:///d:/Masters/2026-Summer%20Semester/OOP/Morse-Code-Decoder/Morse-Code-Decoder/tests/test_corrector.py) | `MorseCorrector` construction, clean text pass-through, `[?]` gap filling (single, multiple, consecutive, start/end positions), confusion-fix conservatism, ham-radio vocabulary preservation, and `_NgramModel` internals (log-prob, Laplace smoothing, vocab) |
| `conftest.py` | Adds the project root to `sys.path` for all tests |

### Other Files

| File | Role |
|------|------|
| `requirements.txt` | Python dependencies: `numpy`, `scipy`, `torch`, `torchaudio`, `pydub`, `matplotlib`, `sounddevice`. Optional: `pyaudiowpatch` (WASAPI system audio on Windows). |
| `data/test/` | Sample WAV files used by the tests. |

---

## DSP Decode Pipeline

The `MorseEngine` decode path, step by step:

1. **Frequency detection** — Welch PSD on the raw audio, peak in 300–1200 Hz range
2. **Bandpass filter** — 5th-order Butterworth, carrier ± 200 Hz
3. **Normalise** — divide by peak absolute value
4. **SNR estimate** — `20 × log10(P90 / P10)` of the envelope; gate at 28 dB
5. **Transition rate check** — transitions/sec on the binary signal; > 150/s = noise, not Morse
6. **Smooth & threshold** — 5 ms moving average; adaptive factor (0.35–0.50) based on SNR
7. **Pulse segmentation** — run-length encode the binary signal into (state, duration) pairs
8. **Noise spike filter** — drop pulses shorter than `rough_unit / 3`
9. **Dit estimate** — gap-based clustering on sorted ON durations to find the dit/dah boundary
10. **Decode** — ON durations: `< unit × 2` → dit, else dah. OFF durations: `> unit × 5` → word space, `> unit × 2.5` → char space
11. **Lookup** — `MORSE_MAP[morse_pattern]` per character; unknown patterns → `[?]`

---

## N-gram Correction Layer

After the DSP pipeline produces raw decoded text (which may contain `[?]` symbols and mis-decoded characters), the `MorseCorrector` applies two probabilistic correction passes:

### Pass 1 — Fill `[?]` Gaps
When the DSP engine encounters an unrecognised Morse pattern, it inserts a `[?]` placeholder. The corrector replaces each `[?]` by scoring every candidate character using **bidirectional context**:
- **Forward model**: probability of the candidate given the characters to its left (trigram context)
- **Backward model**: probability of the candidate given the characters to its right (reversed trigram trained on reversed corpus)

The candidate with the highest combined score wins. Example: `HE[?]LO` → `HELLO`.

### Pass 2 — Fix Morse Confusion Pairs
Morse characters whose patterns differ by a single element (one extra dit or dah) are commonly confused in noisy signals. The corrector checks each character against its known confusion partners:

| Confusion Pair | Pattern Similarity |
|---|---|
| E (`.`) ↔ I (`..`) | 1 vs 2 dots |
| I (`..`) ↔ S (`...`) | 2 vs 3 dots |
| S (`...`) ↔ H (`....`) | 3 vs 4 dots |
| T (`-`) ↔ M (`--`) | 1 vs 2 dashes |
| N (`-.`) ↔ A (`.-`) | dash-dot vs dot-dash |

A swap only happens when the alternative is **≥15× more probable** in the surrounding trigram context — this conservative threshold prevents over-correction.

### Language Model Details
- **Model type**: Interpolated character-level trigram with Laplace smoothing
- **Interpolation weights**: λ₁=0.10 (unigram), λ₂=0.30 (bigram), λ₃=0.60 (trigram)
- **Training corpus**: ~5,000 characters of uppercase English prose, common function words, letter-cluster patterns, numbers, and amateur-radio QSO language (CQ, QTH, RST, callsigns, etc.)
- **Performance**: <1 ms per typical Morse transmission

---

## Dependencies

```bash
pip install numpy scipy torch torchaudio pydub matplotlib sounddevice

# For system audio capture (Windows WASAPI loopback):
pip install pyaudiowpatch
```

Run tests:

```bash
pytest
```

---

## 🌿 Active Experimental Branches

The `main` branch provides the stable, live DSP decoding experience described above. Two additional branches explore further directions:

- **`CNN-LSTM`** — An alternative Machine Learning approach that bypasses the DSP pipeline entirely. Trains a CNN-LSTM neural network on Mel spectrograms with CTC loss to decode Morse characters directly from audio. Includes dataset generation, training scripts, and inference tools. See the README on the `CNN-LSTM` branch for full training instructions.
- **`dsp-ml-approach`** — An experimental staging ground for advanced tuning of the DSP logic and N-gram corrector. Contains work-in-progress enhancements that are not yet stable enough to merge into `main`.
