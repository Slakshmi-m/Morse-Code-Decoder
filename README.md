# Morse Code Decoder — DSP + ML Approach Branch

> ** WORK IN PROGRESS** — This branch (`dsp-ml-approach`) builds on the stable `main` branch by adding an **LLM-based semantic verification layer**, improved DSP timing logic, and accuracy benchmarking tools. These features are still under active development and are **not yet merged into `main`**.

---

## What This Branch Adds (vs `main`)

The `main` branch decodes Morse using a DSP pipeline + N-gram corrector. This branch extends that with three major additions:

### 1. LLM Semantic Verification Layer (`src/semantic_corrector.py`)
A **locally-hosted Ollama LLM** (Mistral, Llama, Phi, etc.) acts as a final verification step on top of the N-gram corrector. The full decode pipeline becomes:

```
Audio → DSP Engine → N-gram Corrector → LLM Semantic Verifier → Final Output
```

**How it works:**
- After the N-gram corrector produces its best text, the `SemanticCorrector` sends both the **decoded text** and the **original Morse dot/dash patterns** to a local Ollama model
- The LLM is prompted as a "Morse code expert" — it knows English, amateur radio Q-codes, callsigns, and common Morse vocabulary
- The LLM can fix remaining character errors that the statistical N-gram model missed, because it understands **semantic meaning** (e.g. it knows "CQ DE W1ABC" is a valid ham radio call)
- A **similarity guard** rejects the LLM output if it diverges more than 25% from the DSP decode, preventing hallucination
- All LLM calls run in **background daemon threads** — the UI is never blocked
- If Ollama is offline or the model isn't pulled, the system gracefully falls back to displaying just the N-gram corrected output

**Requirements:**
```bash
# Install and start Ollama (https://ollama.ai)
ollama serve

# Pull a model (pick one):
ollama pull mistral        # recommended, ~4 GB
ollama pull llama3         # ~4.7 GB
ollama pull phi3:mini      # ~2.4 GB, faster on CPU
```

### 2. Farnsworth Timing Detection (`src/engine.py`)
The DSP engine now handles **Farnsworth-spaced Morse** — a common practice where characters are sent at fast speed but with stretched gaps between them. Without this fix, every character boundary was decoded as a word space, producing output like `H E L L O` instead of `HELLO`.

**What changed:**
- **OFF-duration calibration**: when all ON pulses are the same length (e.g. all dahs), the shortest OFF duration is used as the dit-unit estimate instead
- **Adaptive word-space clustering**: if the median inter-character gap exceeds the standard word-space threshold, the engine switches from fixed multipliers to gap-based clustering to find the real character/word boundary

### 3. Accuracy Benchmark (`scripts/benchmark.py`)
An end-to-end accuracy testing tool that runs the full decoder (DSP + N-gram) over the synthetic dataset and reports:
- **Character Error Rate (CER)** broken down by noise level and WPM
- **Signal failure rate** (percentage of signals the engine couldn't decode)
- Comparison of raw DSP output vs N-gram corrected output

```bash
python -m scripts.benchmark                    # full run
python -m scripts.benchmark --no-corrector     # skip corrector pass
python -m scripts.benchmark --limit 500        # first N samples only
```

### 4. Augmented N-gram Training (`src/corrector.py`)
The N-gram corrector now **dynamically augments its training corpus** by loading target texts from `data/training/pairs.jsonl` (if present), improving correction accuracy on domain-specific vocabulary.

---

## Updated Architecture

```
Audio Input (mic / WAV / system audio)
        │
        ▼
┌───────────────────────────────┐
│    DSP Engine (src/engine.py) │  ← Now with Farnsworth timing detection
│  filter → envelope →          │
│  threshold → timing →         │
│  Morse table lookup           │
└───────────────┬───────────────┘
                │ raw decoded text (may contain [?])
                ▼
┌───────────────────────────────┐
│  N-gram Corrector             │  ← Augmented with pairs.jsonl corpus
│  (src/corrector.py)           │
│  Pass 1: fill [?] gaps       │
│  Pass 2: fix confusion pairs │
└───────────────┬───────────────┘
                │ corrected text
                ▼
┌───────────────────────────────┐
│  LLM Semantic Verifier  🆕    │  ← Ollama (Mistral / Llama / Phi)
│  (src/semantic_corrector.py)  │
│  Prompt with text + patterns  │
│  Similarity guard (≥75%)      │
└───────────────┬───────────────┘
                │ verified text
                ▼
         Displayed in UI
    (4 output panels on right)
```

---

## Updated UI

The right sidebar now has **four output panels** (vs three in `main`):

1. **Decoded Letters** — letter tiles with Morse patterns
2. **Morse Symbols** — raw dot/dash notation
3. **Full Decoded Text** — N-gram corrected output
4. **LLM Verified** 🆕 — semantically verified output from the local LLM (shows model name, status messages, and the final corrected text in teal)

The LLM panel uses a **2.5-second debounce** — it waits for the text to stabilise before sending to the LLM, avoiding redundant calls during live decoding.

---

## File Changes vs `main`

| File | Change | Description |
|------|--------|-------------|
| `src/semantic_corrector.py` | **NEW** | Ollama LLM integration — `SemanticCorrector` class with async verification, model checking, prompt construction, and thread-safe coalescing |
| `src/engine.py` | Modified | Farnsworth timing: OFF-duration calibration (`_calibrate`), adaptive word-space threshold via gap clustering |
| `src/corrector.py` | Modified | Augmented `build()` loads additional training text from `data/training/pairs.jsonl` |
| `src/ui.py` | Modified | 4th "LLM VERIFIED" panel, `SemanticCorrector` integration, debounced LLM triggering, `full_decode()` method on `StreamingDecoder`, similarity guard, `MAX_SEC` increased to 60s |
| `scripts/benchmark.py` | **NEW** | CER benchmark script with noise-level and WPM bucketing, supports `--no-corrector` and `--limit` flags |

---

## Running

```bash
pip install -r requirements.txt

# Start Ollama (required for LLM verification)
ollama serve

# Run the app
python main.py                  # GUI with LLM panel
python main.py myfile.wav       # decode a file
python main.py --mic            # microphone input
```

> **Note:** If Ollama is not running, the app still works — the LLM panel will show "Ollama offline" and the N-gram corrected text is displayed as the final output.

---

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| DSP Engine | ✅ Stable | Enhanced with Farnsworth timing fix |
| N-gram Corrector | ✅ Stable | Augmented with dynamic corpus loading |
| **LLM Semantic Verifier** | **🚧 WIP** | Works with Ollama locally, needs more testing |
| **Benchmark Script** | **🚧 WIP** | Functional, needs expanded test coverage |

---

## Related Branches
- **`main`** — Stable live dashboard with DSP + N-gram correction (no LLM)
- **`CNN-LSTM`** — Alternative neural network approach that bypasses DSP entirely
