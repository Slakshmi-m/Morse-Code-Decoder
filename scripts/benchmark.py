"""
benchmark.py — End-to-end accuracy test over the synthetic dataset.

Reports Character Error Rate (CER) and signal failure rate broken down
by noise level and WPM, with and without the N-gram corrector.

Usage:
    python -m scripts.benchmark                    # full run
    python -m scripts.benchmark --no-corrector     # skip corrector pass
    python -m scripts.benchmark --limit 500        # first N samples only
"""

import argparse
import json
import os
import sys

# ── path setup so we can import from src/ ─────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.engine import decode_wav
from src.corrector import MorseCorrector


# ── Levenshtein CER ───────────────────────────────────────────────────────────

def _cer(ref: str, hyp: str) -> float:
    """Character Error Rate via Levenshtein edit distance."""
    r, h = list(ref), list(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # Wagner-Fischer DP
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r):
        curr = [i + 1] + [0] * len(h)
        for j, hc in enumerate(h):
            curr[j + 1] = min(
                prev[j + 1] + 1,    # deletion
                curr[j] + 1,        # insertion
                prev[j] + (0 if rc == hc else 1),  # substitution
            )
        prev = curr
    return prev[len(h)] / len(r)


# ── Bucketing helpers ──────────────────────────────────────────────────────────

_NOISE_EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, float("inf")]
_NOISE_LABELS = ["0–5%", "5–10%", "10–15%", "15–20%", "20–25%", ">25%"]

_WPM_EDGES  = [0, 12, 22, float("inf")]
_WPM_LABELS = ["slow (≤12)", "medium (13–22)", "fast (≥23)"]


def _noise_bucket(n: float) -> int:
    for i in range(len(_NOISE_EDGES) - 1):
        if _NOISE_EDGES[i] <= n < _NOISE_EDGES[i + 1]:
            return i
    return len(_NOISE_LABELS) - 1


def _wpm_bucket(w: int) -> int:
    for i in range(len(_WPM_EDGES) - 1):
        if _WPM_EDGES[i] < w <= _WPM_EDGES[i + 1]:
            return i
    return len(_WPM_LABELS) - 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Morse decoder on synthetic data")
    parser.add_argument("--no-corrector", action="store_true",
                        help="Disable the N-gram corrector pass")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N samples")
    args = parser.parse_args()

    meta_path  = os.path.join(ROOT, "data", "synthetic", "metadata.json")
    audio_dir  = os.path.join(ROOT, "data", "synthetic", "audio")

    with open(meta_path) as f:
        records = json.load(f)

    if args.limit:
        records = records[: args.limit]

    corrector = None if args.no_corrector else MorseCorrector()

    # Accumulators keyed by (noise_bucket, wpm_bucket)
    # Each cell: [n_total, n_failures, sum_cer_raw, sum_cer_cor]
    NB = len(_NOISE_LABELS)
    WB = len(_WPM_LABELS)
    cells = [[[0, 0, 0.0, 0.0] for _ in range(WB)] for _ in range(NB)]

    total = len(records)
    print(f"\nBenchmarking {total} samples  (corrector={'OFF' if args.no_corrector else 'ON'})\n")

    for idx, rec in enumerate(records):
        # Progress ticker every 100 samples
        if idx % 100 == 0:
            print(f"  {idx:>5}/{total} ...", end="\r", flush=True)

        wav_path = os.path.join(audio_dir, rec["file"])
        ref      = rec["text"].upper().strip()
        nb       = _noise_bucket(rec["noise"])
        wb       = _wpm_bucket(rec["wpm"])
        cell     = cells[nb][wb]
        cell[0] += 1

        raw = decode_wav(wav_path)

        # Detect engine failure (signal-too-weak / no-signal status strings)
        is_failure = raw.startswith("[") and "[?]" not in raw
        if is_failure:
            cell[1] += 1
            cell[2] += 1.0   # count as 100% CER for raw
            cell[3] += 1.0   # and for corrected
            continue

        raw_upper = raw.upper().strip()
        cell[2] += _cer(ref, raw_upper)

        if corrector is not None:
            cor_upper = corrector.correct(raw).upper().strip()
            cell[3] += _cer(ref, cor_upper)
        else:
            cell[3] += _cer(ref, raw_upper)

    print(f"  {total}/{total} ... done\n")

    # ── Report: by noise level ─────────────────────────────────────────────────
    _print_table(
        title="CER by Noise Level",
        row_labels=_NOISE_LABELS,
        row_label_header="Noise",
        cells=cells,
        axis="noise",
        NB=NB, WB=WB,
    )

    # ── Report: by WPM ────────────────────────────────────────────────────────
    _print_table(
        title="CER by WPM",
        row_labels=_WPM_LABELS,
        row_label_header="WPM",
        cells=cells,
        axis="wpm",
        NB=NB, WB=WB,
    )

    # ── Overall summary ────────────────────────────────────────────────────────
    n_tot  = sum(cells[n][w][0] for n in range(NB) for w in range(WB))
    n_fail = sum(cells[n][w][1] for n in range(NB) for w in range(WB))
    cer_r  = sum(cells[n][w][2] for n in range(NB) for w in range(WB))
    cer_c  = sum(cells[n][w][3] for n in range(NB) for w in range(WB))

    print("=" * 62)
    print(f"  Overall  |  samples: {n_tot}  |  failures: {n_fail} ({100*n_fail/max(n_tot,1):.1f}%)")
    if not args.no_corrector:
        print(f"           |  CER raw: {100*cer_r/max(n_tot,1):.1f}%  |  CER corrected: {100*cer_c/max(n_tot,1):.1f}%")
    else:
        print(f"           |  CER: {100*cer_r/max(n_tot,1):.1f}%")
    print("=" * 62)
    print()


def _print_table(title, row_labels, row_label_header, cells, axis, NB, WB):
    print(f"{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")
    col_w = 12

    header = f"  {row_label_header:<14} {'N':>5}  {'Fail%':>6}  {'CER raw':>8}  {'CER+cor':>8}"
    print(header)
    print(f"  {'-'*58}")

    for i, label in enumerate(row_labels):
        if axis == "noise":
            n_tot  = sum(cells[i][w][0] for w in range(WB))
            n_fail = sum(cells[i][w][1] for w in range(WB))
            cer_r  = sum(cells[i][w][2] for w in range(WB))
            cer_c  = sum(cells[i][w][3] for w in range(WB))
        else:  # wpm
            n_tot  = sum(cells[n][i][0] for n in range(NB))
            n_fail = sum(cells[n][i][1] for n in range(NB))
            cer_r  = sum(cells[n][i][2] for n in range(NB))
            cer_c  = sum(cells[n][i][3] for n in range(NB))

        if n_tot == 0:
            continue
        fail_pct = 100 * n_fail / n_tot
        cer_r_pct = 100 * cer_r / n_tot
        cer_c_pct = 100 * cer_c / n_tot
        print(f"  {label:<14} {n_tot:>5}  {fail_pct:>5.1f}%  {cer_r_pct:>7.1f}%  {cer_c_pct:>7.1f}%")

    print()


if __name__ == "__main__":
    main()
