# Morse Code Decoder - DSP & ML Approach Branch 🌱

> **🚧 WORK IN PROGRESS:** This branch (`dsp-ml-approach`) is an active testing ground. The code here contains experimental logic bridging the traditional DSP pipeline with advanced statistical NLP approaches (N-gram corrector enhancements), some of which are still under heavy development and may not be fully functional yet.

A real-time Morse code decoder that evaluates the effectiveness of merging Digital Signal Processing (DSP) logic with Machine Learning text-correction algorithms.

---

## Objective of this Branch

Unlike the stable `main` branch, this branch specializes in finding the breaking points and performance limits of our **probabilistic N-gram correction** logic. 

**Current development focus:**
- Tuning the n-gram statistical boundaries.
- Experimenting with resolving heavily degraded signals where the primary DSP engine yields numerous `[?]` gaps.
- Handling edge cases in character grouping that the standard DSP pipeline incorrectly fragments. 

*Note: Changes on this branch are pushed here isolated from `main` to prevent the live dashboard from breaking while these features are being finalized.*

---

## Branch Architecture

```
Audio Input
    │
    ▼
┌───────────────────────────────┐
│   Core DSP  (src/engine.py)   │  ← Baseline signal translation
│  filter → threshold → timing  │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│   ML Corrector (WIP Focus)    │  ← Advanced N-Gram tuning area
│  Contextual character matrix  │
└───────────────────────────────┘
    │
    ▼
Decoded Text on Screen
```

## Running the Development Pipeline

To run the app specifically for testing the new DSP-ML corrector algorithms:

```bash
pip install -r requirements.txt

# Run the live environment
python main.py

# Test decoding against noisy samples to observe corrector behaviour
python main.py path/to/noisy-sample.wav
```

### Debugging the Corrector
To isolate issues with the DSP and N-gram behavior, you can run the step-by-step trace tool:
```bash
python utils/debug_morse.py
```

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| DSP Engine | Stable | Base pipeline remains identical. |
| **N-gram Corrector (Basic)** | Integrated | The original iteration was merged into `main`. |
| **N-gram Corrector (Advanced)** | **WIP** | Currently debugging logic flow and experimental weights. Do not merge with `main`. |

## Related Approaches
For the parallel project direction that abandons DSP entirely in favor of a neural-network approach (CNN-LSTM on Mel spectrograms), please checkout the `CNN-LSTM` branch.
