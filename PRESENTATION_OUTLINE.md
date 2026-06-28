# Milestone 1 Presentation Outline
## OOP Project: Morse Code Decoder
### ~10–12 slides | 10–15 minutes

---

## SLIDE 1 — Title

**Morse Code Decoder**
An audio-based Morse code decoder using Signal Processing and Machine Learning

> [Your name] | OOP Course | [Date]

---

## SLIDE 2 — What & Why

**What is Morse code?**
- Letters/numbers encoded as short (dot ·) and long (dash —) audio tones
- Still used in aviation, amateur radio, emergency comms

**The problem:**
- Manual decoding requires training
- Real audio is noisy — frequency drifts, static interference, varying speed
- Goal: a system that listens and decodes automatically, even in noisy conditions

> **Say:** "We built a system that takes any Morse audio — from a file, mic, or speaker — and outputs readable text in real time."

---

## SLIDE 3 — Project Overview

**What the system does:**

1. Accepts audio from a WAV file, microphone, or system audio
2. Detects and filters the Morse tone signal
3. Checks signal quality and rejects noise
4. Decodes the on/off pattern into characters
5. Fixes common decoding errors using a language model
6. Shows everything live in a dashboard UI

**Tech stack:** Python · NumPy/SciPy (DSP) · PyTorch (ML) · Tkinter (UI)

---

## SLIDE 4 — System Architecture

**Simple flow:**

```
Audio Source (WAV / Mic / System)
        ↓
   Audio Input Module
   (8 kHz, chunked stream)
        ↓
   DSP Decoder (engine.py)
   Detect tone → Filter → SNR check → Binary signal → Morse → Text
        ↓
   Error Corrector (corrector.py)
   N-gram language model → fix unknown characters
        ↓
   Live Dashboard UI
   Signal plots + decoded text display

   (Parallel) ML Model (CNN-LSTM) ← trained offline on audio dataset
```

**OOP design:** Each box is a separate class — `AudioInput`, `MorseEngine`, `MorseCorrector`, `DecoderUI`. Loosely coupled, independently testable.

---

## SLIDE 5 — Basic Decoding Logic (DSP)

**How the decoder works — 4 steps:**

1. **Find the tone** — Welch PSD scan (300–1200 Hz) to locate the carrier frequency
2. **Filter the signal** — 5th-order Butterworth bandpass filter (±200 Hz around tone)
3. **Noise gate** — check SNR; if too weak, return `"[Signal too weak]"` instead of garbage
4. **Decode** — convert the on/off pattern into dots/dashes → look up character in Morse table

**Timing logic:**
- Estimate "dit" (·) duration from the audio itself → no manual configuration needed
- Gap < 2.5× dit = same character; Gap > 5× dit = new word

> **Say:** "The smart part is that we infer the timing from the audio — so it works at 5 WPM or 35 WPM without any settings change."

---

## SLIDE 6 — Noise Handling

**Assignment requirement:** Handle noisy signals where levels fluctuate or static is present

**Two-condition noise gate:**

| Check | Threshold | Why |
|-------|-----------|-----|
| SNR (signal-to-noise ratio) | > 28 dB | Separates Morse (~50 dB) from noise (~22 dB) |
| Transition rate | < 150/sec | Noise has 150–400 transitions/sec; Morse has 5–40 |

Both conditions must pass — if either fails, the system reports weak signal rather than producing wrong output.

**Adaptive threshold:** When SNR is borderline, the binarisation threshold is relaxed (35% instead of 50%) to still recover the signal.

---

## SLIDE 7 — ML Model (Overview)

**Why ML in addition to DSP?**
DSP needs hand-tuned thresholds; a neural network learns patterns from data.

**Architecture: CNN-LSTM with CTC loss**

```
Mel Spectrogram (audio → image)
    ↓
3× Conv blocks (extract features)
    ↓
Bidirectional LSTM (read time sequence)
    ↓
Character probabilities → CTC decode → Text
```

- **Input:** Log mel spectrogram (64 frequency bands × time)
- **Vocabulary:** 42 characters (A–Z, 0–9, punctuation) + CTC blank
- **CTC loss:** Handles variable-length alignment — no frame-level labels needed

**Status:** Architecture defined, training pipeline ready, awaiting full training run.

---

## SLIDE 8 — Dataset & Training

**Synthetic dataset** (`scripts/generate_dataset.py`):
- 8,000+ audio samples generated with known labels
- Varied WPM (5–35), frequency (600/800 Hz), noise level (0–25%)
- Gives exact ground-truth with no manual labeling

**Real dataset:**
- 180 samples from Morse Code Ninja (professional recordings)
- Multiple speeds and spacing styles

**Combined:** 68,212 samples for training

**Training:**
- 60 epochs, Adam optimizer, ReduceLROnPlateau scheduler
- Best checkpoint saved (lowest validation loss)
- Train/val split: 90/10

---

## SLIDE 9 — Current Status

**What's complete:**

| Component | Status |
|-----------|--------|
| Audio input (WAV / mic / system) | ✅ Done |
| DSP decoder with noise handling | ✅ Done |
| Error corrector (N-gram) | ✅ Done |
| Live dashboard UI | ✅ Done |
| Dataset generation (8,032 synthetic) | ✅ Done |
| Real data pipeline (180 samples) | ✅ Done |
| CNN-LSTM model architecture | ✅ Done |
| Training script | ✅ Done |
| Test suite (8 modules) | ✅ Done |
| Trained model checkpoint | ⏳ Pending |

**Working demo today:** DSP decoder + corrector + full UI

---

## SLIDE 10 — Future Work

**Next steps (Milestone 2+):**

1. **Train the ML model** on the 68K dataset and measure accuracy (Character Error Rate)
2. **Integrate ML into UI** — toggle between DSP and ML decoder
3. **Expand real dataset** — target 1,000+ real recordings
4. **Continuous decoding** — decode character-by-character as it arrives, not window-by-window
5. **Export & packaging** — save session as text, distribute as standalone app

> **Say:** "The architecture was designed from day one so the ML model can plug in as a drop-in replacement for the DSP decoder — no UI changes needed."

---

## SLIDE 11 — Demo

Show live:
1. Open `python main.py` → load a WAV file
2. Point out: oscilloscope, FFT spectrum, waterfall, binary signal panels
3. Show letter tiles and decoded text populating
4. Load a noisy file → show SNR meter drop and noise gate message

*(Fallback: screenshot from `images/dashboard.png`)*

---

## SLIDE 12 — Q&A

**Questions?**

*(Keep the appendix below for anticipated questions)*

---

## Appendix — Likely Questions

**Q: Why DSP AND ML — isn't one enough?**
A: DSP works right now and is explainable. ML will be more accurate with enough data. The dual-track design lets us compare them fairly.

**Q: How do you handle different speeds?**
A: The dit duration is estimated from the audio itself using gap clustering — no manual setting needed. Works from 5 to 35 WPM automatically.

**Q: What if the signal is too noisy?**
A: The two-condition noise gate returns `"[Signal too weak]"` instead of random characters. Graceful degradation was an explicit design goal.

**Q: Why CTC loss for the ML model?**
A: CTC handles variable-length alignment automatically — we only need the final text label, not which audio frame corresponds to which character.
