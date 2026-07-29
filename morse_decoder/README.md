# Morse Code Decoder — dev3.0

A real-time Morse code decoder that listens to audio and converts beeps into text.
This branch combines two approaches: **DSP** (live decoder, always running) and **CNN-LSTM** (neural network, experimental).

---

## How It Works — Big Picture

```
Audio Input (mic / WAV file / system audio)
        │
        ▼
┌───────────────────────────────┐
│   DSP Decoder  (src/)         │  ← Always running, produces live text
│                               │
│  filter → envelope →          │
│  threshold → timing →         │
│  Morse table → corrector      │
└───────────────────────────────┘
        │
        ▼
   Decoded Text on Screen


(Separate, experimental path)

Audio Input
        │
        ▼
┌───────────────────────────────┐
│   CNN-LSTM Model  (ml/)       │  ← Trained offline, tested on WAV files
│                               │
│  spectrogram → CNN →          │
│  LSTM → CTC decode            │
└───────────────────────────────┘
        │
        ▼
   Decoded Text (printed to terminal)
```

---

## The Two Approaches Explained

### 1. DSP — Digital Signal Processing
> The live working decoder. No training needed.

Takes raw audio and applies a chain of mathematical rules:

| Step | What it does |
|---|---|
| Bandpass filter | Keeps only the Morse tone frequency (e.g. 700 Hz), cuts out all noise |
| Envelope detection | Converts the tone into a smooth volume-over-time curve |
| Thresholding | Decides: is the signal ON or OFF at each moment? |
| Timing | Measures pulse lengths — short ON = dot, long ON = dash |
| Morse lookup | Converts dot/dash patterns to letters (e.g. `.-` = A) |
| N-gram corrector | Fixes misread letters using word probability statistics |

No learning involved. Just maths applied to the audio signal.

---

### 2. CNN-LSTM — Machine Learning Model
> Trained separately. Not yet integrated into the live UI.

Learns to decode Morse by studying thousands of audio examples.

| Step | What it does |
|---|---|
| Mel spectrogram | Converts audio into a 2D image (frequency × time) |
| CNN | Scans the image for local patterns (a tone burst = a dit or dah) |
| LSTM | Reads CNN features left-to-right, remembers what came before |
| CTC decode | Aligns output probabilities to characters without needing exact timing |

**CNN** = Convolutional Neural Network — finds patterns in the spectrogram image.
**LSTM** = Long Short-Term Memory — handles sequences, remembers context over time.
**CTC** = Connectionist Temporal Classification — the training loss function that maps frame-by-frame predictions to the final text without needing precise alignment labels.

---

## Project Structure

```
morse_decoder/
├── main.py                        ← Run this to start the app
│
├── src/                           ← DSP decoder (the live working system)
│   ├── engine.py                  ← Core DSP: filter, threshold, timing, decode
│   ├── corrector.py               ← N-gram probabilistic spell corrector
│   ├── audio_input.py             ← Captures mic / system audio / WAV file
│   ├── constants.py               ← Morse code table (A=.-, B=-..., etc.)
│   └── ui.py                      ← Live dashboard (Tkinter + matplotlib panels)
│
├── ml/                            ← CNN-LSTM neural network (experimental)
│   ├── model.py                   ← CNN-LSTM architecture definition
│   ├── train.py                   ← Trains the model, saves model_best.pt
│   └── inference.py               ← Decodes a WAV file using model_best.pt
│
├── data/                          ← Dataset tools
│   ├── generate_dataset.py        ← Generates synthetic Morse training audio
│   └── download_ninja_dataset.py  ← Downloads real Morse Code Ninja audio
│
└── utils/                         ← Developer tools
    ├── debug_morse.py             ← Step-by-step DSP trace for debugging
    └── visualizer.py              ← Plots waveforms and spectrograms
```

---

## Running the Live App (DSP)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python main.py

# Open a specific WAV file
python main.py myfile.wav

# Use microphone input
python main.py --mic
```

---

## CNN-LSTM Training (ML Path)

### Step 1 — Generate training data

**Option A — Synthetic dataset** (fast, no download):
```bash
python -m data.generate_dataset
```
Generates ~8000 clean Morse audio samples with labels. Takes a few minutes.

**Option B — Morse Code Ninja dataset** (real audio, ~150 MB):
```bash
pip install requests pydub
winget install ffmpeg          # Windows only

python -m data.download_ninja_dataset
```
Downloads real Morse practice audio from Morse Code Ninja. Better quality, longer training.

---

### Step 2 — Train the CNN-LSTM

```bash
# Quick test run (10 epochs, ~15-30 min on CPU)
python -m ml.train --epochs 10

# Full training run (60 epochs, 1-3 hours on CPU)
python -m ml.train --epochs 60

# Train using Ninja dataset
python -m ml.train --dataset ninja_dataset --epochs 10

# Train using both datasets combined
python -m ml.train --dataset ninja_dataset --also-synthetic --epochs 10
```

Training prints loss after every epoch. Lower loss = better model.
The best checkpoint is automatically saved to `model_best.pt`.

---

### Step 3 — Test the trained model

```bash
python -m ml.inference myfile.wav
python -m ml.inference myfile.wav model_best.pt
```

---

## What Each File in `ml/` Does

### `ml/model.py` — CNN-LSTM Architecture
Defines the neural network. Three parts:
- **CNN block** — three convolutional layers that scan the mel spectrogram for tone patterns
- **LSTM block** — bidirectional LSTM that reads the sequence and builds context
- **Output head** — linear layer that maps to character probabilities

### `ml/train.py` — Training Loop
- Loads dataset from `metadata.json`
- Converts each audio file to a mel spectrogram
- Feeds spectrograms through the CNN-LSTM
- Computes CTC loss against the correct transcript
- Adjusts model weights to reduce the loss
- Saves the best checkpoint to `model_best.pt`

### `ml/inference.py` — Inference
- Loads `model_best.pt`
- Converts a WAV file to a mel spectrogram
- Runs it through the trained CNN-LSTM
- Returns the decoded text

---

## Feature Status

| Feature | Status |
|---|---|
| Live audio decoding (DSP) | Working |
| Waterfall / oscilloscope / FFT display | Working |
| N-gram corrector | Working |
| Microphone input | Working |
| System audio capture (WASAPI) | Windows only |
| CNN-LSTM training | Available |
| CNN-LSTM integrated into live UI | Not yet |

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `numpy`, `scipy`, `torch`, `torchaudio`, `matplotlib`, `tkinter`

For the Ninja dataset downloader: `requests`, `pydub` + `ffmpeg`
