"""Tests for src/corrector.py — N-gram probabilistic correction layer."""

import pytest
from src.corrector import MorseCorrector, correct


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def corrector():
    return MorseCorrector()


# ── Construction ──────────────────────────────────────────────────────────────

def test_instantiation():
    c = MorseCorrector()
    assert c is not None


def test_module_level_correct_returns_string():
    result = correct("HELLO")
    assert isinstance(result, str)


# ── Pass-through: clean text should not be mangled ────────────────────────────

def test_clean_text_unchanged(corrector):
    # Text with no [?] and no common Morse confusions at the threshold
    assert corrector.correct("HELLO") == "HELLO"


def test_clean_text_with_space(corrector):
    result = corrector.correct("HELLO WORLD")
    assert "HELLO" in result
    assert "WORLD" in result


def test_empty_string(corrector):
    assert corrector.correct("") == ""


def test_single_letter(corrector):
    result = corrector.correct("A")
    assert isinstance(result, str)
    assert len(result) == 1


# ── Status strings are passed through unchanged ───────────────────────────────

def test_status_string_no_signal(corrector):
    msg = "[No Morse Signal Detected]"
    assert corrector.correct(msg) == msg


def test_status_string_weak_signal(corrector):
    msg = "[Signal too weak or noise only (SNR 24.3 dB) — increase volume or reduce background noise]"
    assert corrector.correct(msg) == msg


# ── [?] filling ───────────────────────────────────────────────────────────────

def test_unknown_token_is_replaced(corrector):
    result = corrector.correct("[?]")
    assert result != "[?]"
    assert len(result) == 1


def test_unknown_token_replaced_by_valid_char(corrector):
    result = corrector.correct("[?]")
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?'/-\"@=+$!")
    assert result in valid


def test_hello_with_gap(corrector):
    # [?] at position 2 — strong context on both sides should suggest 'L'
    result = corrector.correct("HE[?]LO")
    assert "[?]" not in result
    assert len(result) == 5


def test_world_with_gap(corrector):
    result = corrector.correct("W[?]RLD")
    assert "[?]" not in result
    assert len(result) == 5


def test_multiple_gaps(corrector):
    # Two independent [?] tokens
    result = corrector.correct("H[?]LLO W[?]RLD")
    assert "[?]" not in result
    assert len(result) == 11   # 5 + space + 5, each [?] replaced by 1 char


def test_consecutive_gaps(corrector):
    # Two adjacent [?] — greedy fill from left
    result = corrector.correct("[?][?]")
    assert "[?]" not in result
    assert len(result) == 2


def test_all_gaps(corrector):
    result = corrector.correct("[?][?][?][?][?]")
    assert "[?]" not in result
    assert len(result) == 5


def test_gap_at_start(corrector):
    result = corrector.correct("[?]ELLO")
    assert "[?]" not in result
    assert len(result) == 5


def test_gap_at_end(corrector):
    result = corrector.correct("HELL[?]")
    assert "[?]" not in result
    assert len(result) == 5


# ── Filled characters are drawn from the valid Morse character set ─────────────

def test_filled_chars_are_valid(corrector):
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?'/-\"@=+$!")
    result = corrector.correct("CQ [?]E W1ABC")
    for ch in result:
        assert ch in valid or ch == " ", f"Unexpected char: {ch!r}"


# ── Confusion correction ──────────────────────────────────────────────────────

def test_confusion_fix_does_not_corrupt_clean_text(corrector):
    # "THE" has no confusions — must stay unchanged
    assert corrector.correct("THE") == "THE"


def test_confusion_fix_does_not_change_non_confusion_chars(corrector):
    # Chars like B, D, F, G, J, L, O, P, Q, R, U, V, X, Y, Z are not in any
    # confusion set — they must survive intact regardless of context
    clean = "FORD ZERO GOLF JULIET"
    result = corrector.correct(clean)
    for ch in "FDJLOQRUVXYZ":
        assert ch in result or ch not in clean


def test_confusion_threshold_is_conservative(corrector):
    # Even if 'E' and 'I' are confused, the corrector should not change 'E'
    # in a word like "SEND" where the context strongly expects 'E'
    result = corrector.correct("SEND")
    # 'E' should remain — it's well supported by the N-gram model
    assert result[1] == "E"


# ── Ham radio vocabulary ──────────────────────────────────────────────────────

def test_cq_call_unchanged(corrector):
    result = corrector.correct("CQ CQ DE W1ABC")
    assert "CQ" in result
    assert "DE" in result


def test_qth_report_unchanged(corrector):
    result = corrector.correct("QTH LONDON RST 599")
    assert "QTH" in result
    assert "599" in result


def test_rst_report_unchanged(corrector):
    assert corrector.correct("RST 599") == "RST 599"


# ── Output type invariants ────────────────────────────────────────────────────

def test_correct_always_returns_str(corrector):
    for s in ["", "A", "[?]", "HELLO WORLD", "CQ DE W1ABC", "[?][?][?]"]:
        assert isinstance(corrector.correct(s), str)


def test_no_unknown_token_in_output(corrector):
    inputs = [
        "HE[?]LO", "[?]ORLD", "HELL[?]", "[?][?][?]",
        "CQ [?]E W1ABC K", "QTH [?]EW ENGLAND",
    ]
    for s in inputs:
        assert "[?]" not in corrector.correct(s), f"[?] remains in: {corrector.correct(s)!r}"


# ── N-gram model internals ────────────────────────────────────────────────────

def test_ngram_model_log_prob_returns_float():
    from src.corrector import _NgramModel
    m = _NgramModel(order=3)
    m.train("THE QUICK BROWN FOX")
    lp = m.log_prob("T", "")
    assert isinstance(lp, float)
    assert lp < 0   # log-probability is always ≤ 0


def test_ngram_model_unseen_char_is_smoothed():
    from src.corrector import _NgramModel
    m = _NgramModel(order=3)
    m.train("AAAA")
    # 'Z' was never seen — Laplace smoothing should still give a finite value
    lp = m.log_prob("Z", "AA")
    assert math.isfinite(lp)


def test_ngram_trained_on_corpus_has_vocab():
    from src.corrector import _NgramModel
    m = _NgramModel(order=3)
    m.train("HELLO WORLD")
    assert len(m._vocab) > 0


import math
