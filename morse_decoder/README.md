# Morse Code Decoder — dev3.0

A real-time Morse code decoder that converts audio beeps into text.  
Built with two pipelines: **DSP** (signal processing, always live) and **CNN-LSTM** (deep learning, the main ML feature).

> **For teammates:** The ML model is the primary deliverable. DSP is included as a working baseline. The UI ties both together.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python main.py

# 3. In the UI — click "Open WAV" to decode a file, or "Microphone" for live input
```

---

## Project Structure

```
morse_decoder/
│
├── main.py                        ← Entry point — run this to launch the app
├── model_best.pt                  ← Trained CNN-LSTM weights (saved after training)
├── test_model.py                  ← Quick 8-word accuracy check on the saved model
├── requirements.txt
├── .gitignore
│
├── dsp/                           ← DSP pipeline (signal processing decoder)
│   ├── constants.py               ← Morse code table  (A=.-, B=-..., etc.)
│   ├── engine.py                  ← Core DSP: bandpass filter → envelope → timing → decode
│   ├── audio_input.py             ← Captures mic / WAV file / system audio
│   └── corrector.py               ← N-gram spell corrector for DSP output
│
├── ui/
│   └── ui.py                      ← Tkinter dashboard (oscilloscope, FFT, waterfall, text tiles)
│
├── ml/                            ← ML pipeline (CNN-LSTM neural network)
│   ├── model.py                   ← Network architecture: Conv1d × 3 + BiLSTM × 2 + CTC head
│   ├── train.py                   ← Training loop — run this to train/retrain
│   ├── inference.py               ← Loads model_best.pt and decodes a WAV file
│   └── data/
│       ├── generate_dataset.py    ← Generates synthetic Morse training audio
│       └── download_ninja_dataset.py  ← (Optional) downloads real Morse audio
│
└── utils/
    ├── debug_morse.py             ← Step-by-step DSP trace for debugging
    └── visualizer.py              ← Waveform and spectrogram plots
```

---

## How the Two Pipelines Work

### Pipeline 1 — DSP (Digital Signal Processing)

> Always running in the background. No training needed. Works in real time.

```
Audio → Bandpass filter → Envelope detection → Threshold → Timing → Morse table → Text
```

| Step | What it does |
|---|---|
| Bandpass filter | Isolates the Morse tone (e.g. 700 Hz), cuts out noise |
| Envelope detection | Converts the tone into a volume-over-time curve |
| Thresholding | Decides ON / OFF at each millisecond |
| Timing | Short ON = dot, long ON = dash; gaps → letter/word boundaries |
| Morse table | `.-` → A, `-...` → B, etc. (defined in `dsp/constants.py`) |
| N-gram corrector | Fixes misread characters using letter-pair statistics |

**File:** `dsp/engine.py`

---

### Pipeline 2 — CNN-LSTM (Machine Learning) ← Main ML Feature

> Trained on synthetic audio. Decodes WAV files using a neural network.

```
WAV file → Mel spectrogram → CNN (feature extractor) → BiLSTM (sequence model) → CTC decode → Text
```

| Step | What it does |
|---|---|
| Mel spectrogram | Converts audio into a 2D frequency-vs-time image (32 mel bins, 8 kHz) |
| Conv1d × 3 | Scans across time to extract local tone/silence patterns |
| Bidirectional LSTM × 2 | Reads the sequence forward and backward, builds context |
| Linear + log-softmax | Maps each time frame to a probability over 38 characters |
| CTC decode | Collapses per-frame predictions to final text (no alignment labels needed) |

**Files:** `ml/model.py`, `ml/train.py`, `ml/inference.py`

**Vocabulary:** A–Z, 0–9, space = 37 characters + 1 CTC blank = 38 total  
**Architecture:** 651,366 trainable parameters  
**Training loss:** CTC (Connectionist Temporal Classification)

---

## Training the CNN-LSTM Model

### Step 1 — Generate the training dataset

```bash
python ml/data/generate_dataset.py
```

Creates `dataset/` with ~3630 synthetic Morse WAV files and a `metadata.json` label file.  
Takes about 1–2 minutes. The dataset is **not committed to git** (too large) — each teammate must generate it locally.

### Step 2 — Train

```bash
python ml/train.py --epochs 35
```

- Trains for 35 epochs on CPU (~5 min/epoch → ~3 hours total)
- Prints loss after each epoch — lower is better
- Automatically saves the best checkpoint to `model_best.pt`
- Runs an 8-word decode test at the end

**What good training looks like:**
```
Epoch   1/35  train=210.4  val=198.3  ← saved
Epoch   5/35  train=12.1   val=9.8    ← saved
Epoch  15/35  train=0.42   val=0.31   ← saved
Epoch  25/35  train=0.08   val=0.04   ← saved   ← good model starts here
```

Target: `val_loss < 0.1` for reliable decoding.

### Step 3 — Check the model

```bash
python test_model.py
```

Runs 8 test words through the saved model and prints a score card:

```
--- Quick decode test ---
  [OK]     expected=E      got='E'
  [OK]     expected=SOS    got='SOS'
  [OK]     expected=HAM    got='HAM'
  ...
  Score: 8/8
  Perfect score — model is ready!
```

A score of 8/8 means the model is working well.

---

## Using the UI

```bash
python main.py               # launch with no file
python main.py myfile.wav    # open and decode a WAV file immediately
python main.py --mic         # start with microphone input
```

**UI layout:**

| Panel | What it shows |
|---|---|
| Left column | Live oscilloscope, FFT, waterfall, binary signal, dit-dah histogram |
| Right column | Decoded letter tiles (large cards), Morse symbol stream, full text box |
| Top bar | Open WAV / Microphone / System Audio / Stop / Clear |

When you open a WAV file:
1. The DSP decoder runs live as the file plays
2. After the file finishes, the ML model (`model_best.pt`) runs on the full file for a more accurate result
3. The decoded text appears in the right panel

---

## Feature Status

| Feature | Status |
|---|---|
| Live DSP decoding | Working |
| CNN-LSTM ML decoding (WAV file) | Working — needs `model_best.pt` |
| ML integrated into UI | Working |
| Oscilloscope / FFT / waterfall display | Working |
| N-gram corrector | Working |
| Microphone input | Working |
| System audio capture (WASAPI) | Windows only |
| Spaces between words in ML output | Needs retrain with multi-word dataset |

---

## Dependencies

```bash
pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| `numpy`, `scipy` | DSP signal processing |
| `matplotlib` | Live plots in UI |
| `torch`, `torchaudio` | CNN-LSTM model and mel spectrogram |
| `sounddevice` | Microphone input (cross-platform) |
| `pyaudiowpatch` | System audio loopback (Windows / WASAPI) |

> `tkinter` is part of Python's standard library. If missing on Linux: `sudo apt-get install python3-tk`

---

## For Teammates — What To Do First

1. **Clone the repo and install dependencies:**
   ```bash
   git clone https://github.com/kpnair99/kern.git
   cd kern/morse_decoder
   pip install -r requirements.txt
   ```

2. **Generate the dataset** (required before training):
   ```bash
   python ml/data/generate_dataset.py
   ```

3. **Train the model** (or use the committed `model_best.pt` if already present):
   ```bash
   python ml/train.py --epochs 35
   ```

4. **Check the model:**
   ```bash
   python test_model.py
   ```

5. **Run the app:**
   ```bash
   python main.py
   ```

---

## Known Issues

- **No spaces between words** — the current dataset only has single-word samples, so the ML model never outputs a space character. Fix: retrain with multi-word phrases in the dataset.
- **Rare-frequency WAV files** — WAV files recorded at unusual tone frequencies (below 500 Hz) may decode with lower accuracy. The training covers 500–1200 Hz.
- **CPU training is slow** — ~5 minutes per epoch on CPU. No GPU required, just patience.
