"""
semantic_corrector.py — LLM-based semantic verification layer
=============================================================
Uses a locally-hosted Ollama LLM (Mistral, Llama, etc.) to sanity-check
the output produced by the DSP decoder + N-gram corrector.

The LLM receives:
  1. The Morse dot/dash patterns derived from the decoded characters
  2. The n-gram-corrected text (best automatic interpretation so far)

And returns the most likely intended message, grounded in its knowledge
of English, amateur radio Q-codes, Bible verses, etc.

Requirements
------------
  Ollama running locally  →  https://ollama.ai
  At least one model pulled, e.g.:
      ollama pull mistral          (recommended, ~4 GB)
      ollama pull llama3           (~4.7 GB)
      ollama pull phi3:mini        (~2.4 GB, faster on CPU)

This layer is fully optional: if Ollama is unreachable the n-gram
corrected text is simply shown unchanged.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Callable, Optional


class SemanticCorrector:
    """
    Sends decoded Morse text to a local Ollama LLM for semantic verification.

    All network calls run in daemon threads — the UI is never blocked.
    Concurrent requests are coalesced: a new call while one is in flight
    is silently dropped (the in-flight result arrives shortly anyway).
    """

    DEFAULT_MODEL = "mistral"
    OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
    OLLAMA_TAGS_URL     = "http://localhost:11434/api/tags"
    TIMEOUT_S           = 60

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model      = model
        self._lock      = threading.Lock()
        self._busy      = False
        self._available: Optional[bool] = None  # None = not yet probed

    # ── public API ────────────────────────────────────────────────────────────

    def verify_async(
        self,
        decoded_text: str,
        morse_symbols: str,
        on_result: Callable[[str], None],
        on_error:  Callable[[str], None],
    ) -> None:
        """
        Non-blocking.  Calls ``on_result(corrected)`` or ``on_error(msg)``
        from a background daemon thread when the LLM responds.

        Silently returns without doing anything if a call is already in flight.
        """
        with self._lock:
            if self._busy:
                return
            self._busy = True

        threading.Thread(
            target=self._run,
            args=(decoded_text, morse_symbols, on_result, on_error),
            daemon=True,
        ).start()

    # ── internals ─────────────────────────────────────────────────────────────

    def _run(
        self,
        decoded_text: str,
        morse_symbols: str,
        on_result: Callable[[str], None],
        on_error:  Callable[[str], None],
    ) -> None:
        try:
            # Fast pre-check: is Ollama up and is our model pulled?
            err = self._check_model()
            if err:
                on_error(err)
                return

            payload = json.dumps({
                "model":  self.model,
                "prompt": self._build_prompt(decoded_text, morse_symbols),
                "stream": False,
                "options": {
                    "temperature": 0.05,   # near-deterministic
                    "num_predict": 256,
                },
            }).encode()

            req = urllib.request.Request(
                self.OLLAMA_GENERATE_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_S) as resp:
                data      = json.loads(resp.read())
                corrected = data.get("response", "").strip().strip('"\'')
                if corrected:
                    self._available = True
                    on_result(corrected)
                else:
                    on_error("LLM returned an empty response")

        except urllib.error.URLError:
            self._available = False
            on_error(
                "Ollama offline — run in a terminal:\n"
                "  ollama serve"
            )
        except Exception as exc:
            on_error(f"LLM error: {exc}")
        finally:
            with self._lock:
                self._busy = False

    def _check_model(self) -> Optional[str]:
        """
        Returns an error string if Ollama is unreachable or the model is not
        pulled, otherwise returns None (all good).  Times out in 3 s so the UI
        gets quick feedback instead of hanging for a minute.
        """
        try:
            with urllib.request.urlopen(self.OLLAMA_TAGS_URL, timeout=3) as resp:
                data   = json.loads(resp.read())
                models = [m["name"].split(":")[0] for m in data.get("models", [])]
        except urllib.error.URLError:
            self._available = False
            return (
                "Ollama offline — run in a terminal:\n"
                "  ollama serve"
            )
        except Exception as exc:
            return f"Ollama check failed: {exc}"

        if not models:
            return (
                f"No models pulled yet — run in a terminal:\n"
                f"  ollama pull {self.model}\n"
                f"(phi3:mini is faster on CPU; mistral gives better results)"
            )

        base = self.model.split(":")[0]
        if base not in models:
            available = ", ".join(models)
            return (
                f"Model '{self.model}' not found.\n"
                f"Available: {available}\n"
                f"Or pull it: ollama pull {self.model}"
            )

        return None  # model is ready

    @staticmethod
    def _build_prompt(decoded_text: str, morse_symbols: str) -> str:
        return (
            "You are a Morse code expert assistant.\n\n"
            "An audio Morse code signal was decoded automatically. "
            "The decoded text is usually correct; only rarely does audio noise "
            "cause a single character to be misread.\n\n"
            "Detected Morse dot/dash patterns (from the audio signal):\n"
            f"  {morse_symbols}\n\n"
            "Automated decode result:\n"
            f"  {decoded_text}\n\n"
            "The transmission is likely one of:\n"
            "  • Amateur radio (CQ calls, RST signal reports, callsigns like W1ABC, "
            "Q-codes such as QSL / QTH / QRZ, 73, de)\n"
            "  • Plain English text (e.g. a Bible verse, general message, "
            "sentence, or phrase)\n"
            "  • Numbers, abbreviations, or a mix of the above\n\n"
            "Rules (follow strictly):\n"
            "  1. Output ONLY the text — no explanation, no preamble, "
            "no trailing commentary.\n"
            "  2. Preserve word spacing and any numbers exactly.\n"
            "  3. Use ALL CAPS (Morse decodes to uppercase).\n"
            "  4. CRITICAL: If the decoded text already looks correct — valid words, "
            "callsigns, Q-codes, abbreviations, or numbers — return it EXACTLY "
            "unchanged, character for character.  Do NOT rephrase, summarise, "
            "or creatively reinterpret.\n"
            "  5. Only change a character when you are certain it is wrong based on "
            "the dot/dash pattern AND surrounding context.  Make the minimum number "
            "of single-character substitutions necessary — never more than two.\n\n"
            "Corrected text:"
        )
