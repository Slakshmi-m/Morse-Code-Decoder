# Morse Code Decoder - Live Dashboard

A real-time Morse code decoder with a live GUI dashboard. Audio is captured from a WAV file, microphone, or system speakers and decoded to text using a DSP signal processing pipeline. Results are displayed in a 5 - panel signal visualiser with letter tiles and a live text transcript.

---

## Features

- **Three audio sources**: WAV file playback, microphone (via `sounddevice`), or system audio (WASAPI loopback on Windows via `pyaudiowpatch`)
- **Adaptive noise handling**: SNR-gated decoder rejects pure noise; adaptive threshold recovers weak signals
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
│   DecoderUI ─► LivePlotter (5 panels)                   │
│             ─► LetterTilePanel                           │
│             ─► StreamingDecoder (coordinator)            │
└────┬─────────────────────────────────┬────────────────────┘
     │                                 │
     ▼                                 ▼
┌─────────────┐              ┌──────────────────────┐
│src/audio_   │              │     src/engine.py    │
│input.py     │──samples──►  │     MorseEngine      │
│AudioInput   │              │  (DSP decode)        │
│ - WAV file  │              └──────────┬───────────┘
│ - Mic (sd)  │                         │ decoded text
│ - WASAPI    │                         ▼
└─────────────┘                  displayed in UI

src/constants.py ─────────────────────► used by engine + ui
```

---

## File Reference

### Entry Point

| File | Role |
|------|------|
| [main.py](main.py) | Parses CLI args (`--mic`, WAV path), checks tkinter availability, launches `DecoderUI`. |

### `src/` — Core Library

| File | Role |
|------|------|
| [src/constants.py](src/constants.py) | Defines `MORSE_MAP` (Morse pattern → character) and `TEXT_TO_MORSE` (inverse). Single source of truth for the symbol table — both `engine.py` and `ui.py` import from here. |
| [src/engine.py](src/engine.py) | `MorseEngine`: the primary DSP decoder. Pipeline: auto-detect carrier frequency via Welch PSD → bandpass filter (±200 Hz) → amplitude envelope → adaptive binary threshold → pulse duration segmentation → dit/dah/space classification → `MORSE_MAP` lookup. Includes an SNR gate (28 dB threshold) and a transition-rate check that together reject pure background noise. Also exposes `decode_wav()` as a standalone helper. |
| [src/audio_input.py](src/audio_input.py) | `AudioInput`: thread-based audio source with a callback/observer interface. Supports WAV file streaming (`stream_file_async`), microphone via `sounddevice`, and system audio via `pyaudiowpatch` WASAPI loopback. Handles resampling to 8 kHz and stereo→mono conversion internally. |
| [src/ui.py](src/ui.py) | The complete GUI. `DecoderUI` builds the window and toolbar. `StreamingDecoder` buffers incoming audio chunks, throttles decode calls to every 0.4 s, and routes results to the UI via a thread-safe queue. `LivePlotter` drives the 5-panel Matplotlib canvas at 150 ms refresh. `LetterTilePanel` renders decoded characters as coloured cards. Uses the Catppuccin Mocha colour palette. |
| [src/\_\_init\_\_.py](src/__init__.py) | Makes `src/` a Python package. |

### Tests

| File | What it tests |
|------|---------------|
| [tests/test_constants.py](tests/test_constants.py) | `MORSE_MAP` completeness (26 letters, 10 digits, special characters) and `TEXT_TO_MORSE` invertibility |
| [tests/test_engine.py](tests/test_engine.py) | `MorseEngine` frequency detection, SNR estimation, and decoding on clean/noisy/silent signals; `decode_wav()` helper |
| [tests/test_audio_input.py](tests/test_audio_input.py) | `AudioInput` initialisation, callback dispatch, resampling, and WAV file streaming |
| [conftest.py](conftest.py) | Adds the project root to `sys.path` for all tests |

### Other Files

| File | Role |
|------|------|
| [requirements.txt](requirements.txt) | Python dependencies: `numpy`, `scipy`, `torch`, `torchaudio`, `pydub`, `matplotlib`, `sounddevice`. Optional: `pyaudiowpatch` (WASAPI system audio on Windows). |
| [data/test/](data/test/) | Sample WAV files used by the tests. |

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

## Future Work

The following features are planned but not yet implemented:

- **Probabilistic correction layer** (`src/corrector.py`) — a character-level N-gram language model (bigram/trigram) that fills `[?]` gaps and corrects common Morse confusions using an English + amateur-radio corpus.
- **CNN-LSTM deep learning model** (`src/model.py`) — an end-to-end neural decoder using mel spectrograms as input and CTC loss for sequence alignment, as an alternative to the DSP pipeline.
- **Synthetic dataset generation** (`scripts/generate_dataset.py`) — renders labelled Morse audio at randomised speeds, frequencies, noise levels, and Farnsworth spacing.
- **Real-data ingestion pipeline** (`scripts/prepare_real_data.py`, `scripts/download_ninja.py`) — ingest real-world Morse recordings (e.g. Morse Code Ninja) and produce a training-compatible `metadata.json`.
- **Dataset mixing** (`scripts/mix_datasets.py`) — merge synthetic and real datasets with configurable oversampling ratios.
- **Model training & inference scripts** (`scripts/train.py`, `scripts/inference.py`) — train the CNN-LSTM from a `metadata.json` manifest and run batch offline decoding.
- **macOS / Linux system audio capture** — extend system audio beyond Windows WASAPI to CoreAudio (macOS) and PulseAudio/PipeWire (Linux).
- **Decoded text export** — save the live transcript to a `.txt` file from the GUI.
- **Broader audio format support** — accept MP3, FLAC, and OGG in addition to WAV via `pydub`.
