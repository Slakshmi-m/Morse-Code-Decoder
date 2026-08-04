# Morse Code Decoder — dev3.0

A real-time Morse code decoder that converts audio into text using two cooperating
systems: **DSP** (Digital Signal Processing) and a **CNN-LSTM** neural network.
The key design is that DSP does not compete with ML — DSP **feeds** ML by cleaning
the signal first, then ML does the character recognition.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install MP3 support
pip install pydub        # also needs ffmpeg: https://ffmpeg.org/download.html

# 3. Run the app
python main.py

# 4. Click "Open Audio" to load a WAV or MP3, or click "Microphone" for live input
```

---

## Project Structure

```
morse_decoder/
│
├── main.py                     ← Entry point — run this to launch the app
├── model_best.pt               ← Trained CNN-LSTM weights (saved after training)
├── requirements.txt
├── .gitignore
│
├── dsp/                        ← DSP pipeline
│   ├── constants.py            ← Morse table (A=.-, B=-..., etc.) + ALL_CHARS
│   ├── engine.py               ← Core DSP: carrier detect → bandpass → envelope → decode
│   ├── audio_input.py          ← Captures mic / WAV / MP3 / system audio
│   └── corrector.py            ← N-gram spell corrector for DSP output
│
├── ui/
│   └── ui.py                   ← Tkinter dashboard (oscilloscope, FFT, waterfall, tiles)
│
└── ml/                         ← ML pipeline
    ├── model.py                ← Network: Conv1d × 3 + BiLSTM × 2 + CTC head
    ├── train.py                ← Training loop
    ├── inference.py            ← DSP preprocessing + model inference
    └── data/
        └── generate_dataset.py ← Generates synthetic Morse training audio
```

---

## Full Pipeline — Audio Input to Decoded Text

### Mode 1 — DSP Only

> Always live. No model needed. Used as the real-time preview in ML mode too.

```
┌─────────────────────────────────────────────────────────┐
│  AUDIO INPUT                                            │
│  WAV / MP3 / Microphone / System Audio                  │
│  Resampled to 8 000 Hz mono int16                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1 — Carrier Detection  (dsp/engine.py)            │
│                                                         │
│  Welch PSD on the raw buffer                            │
│  Finds the dominant peak in 300–1200 Hz                 │
│  Example: 748 Hz, 1070 Hz, 700 Hz                       │
└────────────────────────┬────────────────────────────────┘
                         │  detected_freq
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2 — Bandpass Filter                               │
│                                                         │
│  Butterworth order-5, ±200 Hz around carrier            │
│  Example: 748 Hz → keeps 548–948 Hz, kills all else     │
│  Removes wideband noise, music, background              │
└────────────────────────┬────────────────────────────────┘
                         │  filtered signal
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3 — Envelope + Normalise                          │
│                                                         │
│  abs(filtered) → normalise peak to 1.0                  │
│  10 ms box smoothing to remove sub-Morse noise spikes   │
│  Result: smooth volume curve, 0.0 (silence) to 1.0 (on) │
└────────────────────────┬────────────────────────────────┘
                         │  smooth envelope
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4 — SNR Gate                                      │
│                                                         │
│  SNR = 20 · log10(P90 / P10) of the envelope            │
│  Pure noise  ≈ 22 dB                                    │
│  Noisy Morse ≈ 33 dB                                    │
│  Clean Morse ≈ 50–140 dB                                │
│  Gate at 28 dB + transition-rate check (< 150/s)        │
│  → rejects pure noise before attempting decode          │
└────────────────────────┬────────────────────────────────┘
                         │  passed gate
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5 — Adaptive Threshold → Binary ON/OFF            │
│                                                         │
│  threshold = floor + (peak − floor) × factor            │
│    SNR ≥ 15 dB → factor = 0.50 (clean)                  │
│    SNR ≥  8 dB → factor = 0.42 (moderate noise)         │
│    else        → factor = 0.35 (heavy noise)            │
│                                                         │
│  Output: 1 = tone ON  (dit or dah)                      │
│          0 = tone OFF (gap)                             │
└────────────────────────┬────────────────────────────────┘
                         │  binary signal
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 6 — Noise Spike Filter                            │
│                                                         │
│  Run-length encode binary → [(state, duration), ...]    │
│  rough_unit = 25th percentile of all ON durations       │
│  Drop any segment shorter than max(10ms, rough_unit/3)  │
│  → removes music/background artefacts < 10ms            │
│    (real Morse is ≥ 10ms even at 120 WPM)               │
└────────────────────────┬────────────────────────────────┘
                         │  clean segments
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 7 — Variable Speed Detection                      │
│                                                         │
│  _estimate_dit(): gap-cluster lower 80% of ON durs      │
│    → finds the valley between dits and dahs             │
│    → unit = mean of dit-class durations                 │
│                                                         │
│  _estimate_space_thresholds():                          │
│    char_thresh = unit × 2.0  (fixed, between 1× and 3×) │
│    word_thresh = cluster OFF gaps > char_thresh          │
│    → adaptive to any WPM (15–60 WPM tested)             │
└────────────────────────┬────────────────────────────────┘
                         │  unit, char_thresh, word_thresh
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 8 — Morse Decode                                  │
│                                                         │
│  ON  duration < unit×2 → dit  "."                       │
│  ON  duration ≥ unit×2 → dah  "-"                       │
│  OFF > word_thresh     → flush char + append space       │
│  OFF > char_thresh     → flush char                     │
│                                                         │
│  MORSE_MAP lookup: ".-" → A, "-..." → B, etc.           │
│  Unknown pattern → [?] placeholder                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
                   Decoded Text
```

---

### Mode 2 — DSP + CNN-LSTM (ML Mode)

> DSP cleans and segments the signal. ML does the character recognition.
> DSP is not replaced — it feeds the ML model.

```
┌─────────────────────────────────────────────────────────┐
│  AUDIO INPUT  (same as DSP mode)                        │
│  WAV / MP3 / Microphone / System Audio → 8 000 Hz mono  │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
    ┌──────────────────┐   ┌─────────────────────────────┐
    │ DSP LIVE PREVIEW │   │  ML PIPELINE (after file    │
    │ (Steps 1–8 above)│   │  finishes or every cycle    │
    │ Updates tiles    │   │  for live audio)             │
    │ every 250ms      │   └──────────────┬──────────────┘
    └──────────────────┘                  │
                                          ▼
              ┌───────────────────────────────────────────┐
              │  STAGE A — DSP Word Splitting             │
              │  _split_at_word_gaps()                    │
              │                                           │
              │  1. Welch PSD → carrier (400–1200 Hz)     │
              │  2. Bandpass ±200 Hz                      │
              │  3. Envelope → 10ms smooth → binary       │
              │  4. Filter spikes < 10ms                  │
              │  5. rough_unit from clean ON durations    │
              │  6. word_thresh = rough_unit × 5.0        │
              │  7. Find OFF gaps ≥ word_thresh            │
              │     → split buffer at gap centres         │
              │                                           │
              │  Output: [THE] [LORD] [IS] [MY] ...       │
              └──────────────────┬────────────────────────┘
                                 │ one chunk per word
                                 ▼
              ┌───────────────────────────────────────────┐
              │  STAGE B — DSP Signal Reconstruction      │
              │  _dsp_preprocess()  — per word chunk      │
              │                                           │
              │  1. Welch PSD → carrier                   │
              │  2. Bandpass ±200 Hz                      │
              │  3. Envelope → normalise                  │
              │  4. 10ms smooth                           │
              │  5. Threshold → binary ON/OFF             │
              │  6. Filter spikes < 10ms                  │
              │  7. Reconstruct at 700 Hz sine wave:      │
              │                                           │
              │  Original (748 Hz, noisy):                │
              │  ~~▲~~▲▲▲~~▲~~▲▲▲~~▲▲▲▲▲▲▲~~             │
              │                                           │
              │  Reconstructed (700 Hz, clean):           │
              │   ▲  ▲▲▲  ▲  ▲▲▲  ▲▲▲▲▲▲▲               │
              │  dit dah dit dah    dah                   │
              │                                           │
              │  Always 700 Hz regardless of original     │
              │  carrier → matches training data exactly  │
              └──────────────────┬────────────────────────┘
                                 │ clean 700 Hz signal
                                 ▼
              ┌───────────────────────────────────────────┐
              │  STAGE C — Silence Trimming               │
              │  _trim_silence()                          │
              │                                           │
              │  Remove leading/trailing silence padding  │
              │  from word gap splits                     │
              │  Prevents model seeing silence as signal  │
              │  (silence was being decoded as "5" = ·····)│
              └──────────────────┬────────────────────────┘
                                 │ tight word signal only
                                 ▼
              ┌───────────────────────────────────────────┐
              │  STAGE D — Mel Spectrogram                │
              │                                           │
              │  sample_rate = 8 000 Hz                   │
              │  n_fft       = 256                        │
              │  hop_length  = 64  (8 ms steps)           │
              │  n_mels      = 32                         │
              │  AmplitudeToDB → log scale                │
              │                                           │
              │  Output shape: [1, 32, time_frames]       │
              │  Each column = 8ms snapshot of spectrum   │
              └──────────────────┬────────────────────────┘
                                 │ mel spectrogram tensor
                                 ▼
              ┌───────────────────────────────────────────┐
              │  STAGE E — CNN Feature Extraction         │
              │  3 × Conv1d layers  (ml/model.py)         │
              │                                           │
              │  Conv1d(32→64, k=3)  + BatchNorm + ReLU  │
              │  Conv1d(64→64, k=3)  + BatchNorm + ReLU  │
              │  Conv1d(64→64, k=3)  + BatchNorm + ReLU  │
              │                                           │
              │  Scans across time frames                 │
              │  Learns: dit shape, dah shape,            │
              │          silence shape, tone boundaries   │
              └──────────────────┬────────────────────────┘
                                 │ feature sequence
                                 ▼
              ┌───────────────────────────────────────────┐
              │  STAGE F — BiLSTM Sequence Model          │
              │  2 × Bidirectional LSTM  (hidden=128)     │
              │                                           │
              │  Reads LEFT → RIGHT and RIGHT → LEFT      │
              │  Understands timing patterns across the   │
              │  whole word, not just local features      │
              │                                           │
              │  Learns: dit→dah ratios (speed-invariant) │
              │          character boundaries             │
              │          full symbol sequences            │
              └──────────────────┬────────────────────────┘
                                 │ per-frame character probabilities
                                 ▼
              ┌───────────────────────────────────────────┐
              │  STAGE G — CTC Greedy Decode              │
              │                                           │
              │  Linear layer → 38 classes                │
              │  (A–Z, 0–9, space, blank)                 │
              │  log_softmax → per-frame log probs        │
              │                                           │
              │  Greedy: argmax at each frame             │
              │  Collapse: LLOORRDD → LORD                │
              │  Remove blanks → "LORD"                   │
              └──────────────────┬────────────────────────┘
                                 │ word result
                                 ▼
              ┌───────────────────────────────────────────┐
              │  Join all word results with spaces        │
              │  "THE" + "LORD" + "IS" + ... →            │
              │  "THE LORD IS MY SHEPHERD"                │
              └───────────────────────────────────────────┘
```

---

## Why DSP Feeds ML (Not Replaces It)

| What DSP does for ML | Why it matters |
|---|---|
| Detects real carrier frequency | Model never sees wrong frequency band |
| Bandpass filter removes noise | Model receives clean signal, not noisy raw audio |
| Reconstructs at fixed 700 Hz | Training data was 400–1200 Hz; 700 Hz is the centre — model sees in-distribution input every time |
| Filters spikes < 10 ms | Background music creates 2–5 ms artefacts; real Morse is always ≥ 10 ms |
| Splits at word boundaries | Model was trained on single words; splitting gives it the same input format |
| Trims silence padding | Prevents silence between words being decoded as "5" (·····) |

---

## Model Architecture

```
Input: Mel spectrogram  [batch, 1, n_mels=32, time_frames]
  │
  ├── Conv1d(32 → 64, k=3) + BatchNorm + ReLU + Dropout(0.2)
  ├── Conv1d(64 → 64, k=3) + BatchNorm + ReLU + Dropout(0.2)
  ├── Conv1d(64 → 64, k=3) + BatchNorm + ReLU + Dropout(0.2)
  │
  ├── Reshape → [batch, time_frames, 64]
  │
  ├── BiLSTM(64 → 128, 2 layers, dropout=0.3)
  │
  └── Linear(256 → 38) + log_softmax
        38 = A–Z (26) + 0–9 (10) + space (1) + CTC blank (1)

Total parameters: 651,366
Loss function:    CTC (Connectionist Temporal Classification)
Optimizer:        Adam, lr=1e-3 with ReduceLROnPlateau
```

---

## Training

### Step 1 — Generate training data

```bash
python ml/data/generate_dataset.py
```

Creates `dataset/` with synthetic Morse WAV files.

Coverage:
- All 42 characters × 5 speeds (15/20/25/30/35 WPM) × 3 frequencies (600/800/1070 Hz) — guaranteed
- 3000 random samples: varied WPM, frequency (400–1200 Hz), noise (0–25%)

> Dataset is not committed to git — each teammate runs this locally.

### Step 2 — Train

```bash
python ml/train.py --epochs 35
```

- Saves best checkpoint to `model_best.pt` automatically
- Target: `val_loss < 0.1`

**Sample training progress:**
```
Epoch   1/35  train=210.4  val=198.3
Epoch   5/35  train=12.1   val=9.8
Epoch  15/35  train=0.42   val=0.31
Epoch  25/35  train=0.08   val=0.04   ← model ready
```

---

## Using the App

```bash
python main.py               # launch
python main.py myfile.wav    # open file immediately
python main.py --mic         # start with microphone
```

### UI Layout

| Panel | What it shows |
|---|---|
| ① Oscilloscope | Raw waveform — ON/OFF pattern visible |
| ② FFT Spectrum | Frequency peaks — Morse carrier marked in yellow |
| ③ Waterfall | Time vs frequency — Morse appears as a bright vertical stripe |
| ④ Binary Signal | Thresholded ON/OFF — should show clean rectangular pulses |
| ⑤ Dit/Dah Histogram | Duration distribution — two clear clusters = dits and dahs |
| Right: Decoded Letters | Last 8 decoded characters as large coloured tiles |
| Right: Morse Symbols | Raw dot-dash stream |
| Right: Full Decoded Text | Final decoded sentence |

### Mode Switch (top right)

| Mode | Behaviour |
|---|---|
| **DSP** | Real-time decode every 250ms from rolling 6s buffer. Fast, always live. |
| **ML** | DSP runs live for immediate tile feedback. When file finishes (or every cycle for mic), full DSP+CNN-LSTM pipeline runs and replaces with more accurate result. |

### Supported Inputs

| Input | How to use |
|---|---|
| WAV file | Click "Open Audio" → select .wav |
| MP3 file | Click "Open Audio" → select .mp3 (needs `pydub` + ffmpeg) |
| Microphone | Click "Microphone" |
| System audio | Click "System Audio" (Windows WASAPI — captures speakers) |

---

## Feature Status

| Feature | Status |
|---|---|
| Live DSP decoding | Working |
| CNN-LSTM ML decoding | Working — needs `model_best.pt` |
| DSP → ML integrated pipeline | Working |
| Word gap splitting (DSP-guided) | Working |
| MP3 file support | Working — needs `pip install pydub` |
| Carrier auto-detection (400–1200 Hz) | Working |
| Background noise / music rejection | Working — 10ms spike filter |
| Variable speed detection (15–60 WPM) | Working — adaptive timing |
| Oscilloscope / FFT / waterfall / histogram | Working |
| N-gram corrector (DSP mode) | Working |
| Microphone input | Working |
| System audio capture | Windows only (WASAPI) |

---

## Dependencies

```bash
pip install -r requirements.txt
pip install pydub   # optional — MP3 support only
```

| Package | Used for |
|---|---|
| `numpy`, `scipy` | DSP: filtering, PSD, envelope |
| `matplotlib` | Live plots in UI |
| `torch`, `torchaudio` | CNN-LSTM model and mel spectrogram |
| `sounddevice` | Microphone input |
| `pyaudiowpatch` | System audio loopback (Windows) |
| `pydub` | MP3 decoding (optional) |

> `tkinter` ships with Python. If missing on Linux: `sudo apt-get install python3-tk`

---

## Teammate Setup

```bash
# 1. Clone
git clone https://github.com/kpnair99/morse-code-decoder.git
cd morse-code-decoder
git checkout dev3.0

# 2. Install
pip install -r requirements.txt

# 3. Generate dataset
python ml/data/generate_dataset.py

# 4. Train
python ml/train.py --epochs 35

# 5. Run
python main.py
```

If `model_best.pt` is already committed, skip steps 3 and 4.

---

## Known Limitations

- **Background music in audio** — Files with music mixed into the Morse signal may produce character errors. The DSP bandpass and 10ms spike filter reduce this, but strong music in the same frequency band as the carrier will still affect accuracy.
- **CPU training is slow** — ~5 min/epoch on CPU, ~3 hours for 35 epochs. No GPU required.
- **System audio (Windows only)** — WASAPI loopback requires `pyaudiowpatch` and is Windows-specific.
