import unittest
from src.corrector import MorseCorrector


class TestMorseCorrector(unittest.TestCase):

    def setUp(self):
        self.corrector = MorseCorrector()

    # ── stats ─────────────────────────────────────────────────────────────────

    def test_stats_contains_expected_keys(self):
        s = self.corrector.stats()
        self.assertIn('MorseCorrector', s)
        self.assertIn('unigrams', s)
        self.assertIn('bigrams', s)
        self.assertIn('trigrams', s)

    def test_ngram_tables_populated(self):
        self.assertGreater(len(self.corrector._unigram), 0)
        self.assertGreater(len(self.corrector._bigram),  0)
        self.assertGreater(len(self.corrector._trigram), 0)

    # ── passthrough cases ─────────────────────────────────────────────────────

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.corrector.correct(''), '')

    def test_signal_error_message_passed_through(self):
        msg = '[Signal too weak — adjust volume]'
        self.assertEqual(self.corrector.correct(msg), msg)

    def test_no_morse_detected_passed_through(self):
        msg = '[No Morse Signal Detected]'
        self.assertEqual(self.corrector.correct(msg), msg)

    def test_clean_input_returns_string(self):
        result = self.corrector.correct('HELLO WORLD')
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    # ── unknown token filling ─────────────────────────────────────────────────

    def test_fills_single_unknown_token(self):
        result = self.corrector.correct('TH[?] QUICK')
        self.assertNotIn('[?]', result)
        self.assertEqual(len(result), len('TH[?] QUICK') - 2)  # [?] → 1 char

    def test_fills_multiple_unknown_tokens(self):
        result = self.corrector.correct('H[?]LL[?] W[?]RLD')
        self.assertNotIn('[?]', result)

    def test_unknown_in_middle(self):
        # [?] embedded mid-string (not at position 0) gets filled
        result = self.corrector.correct('A[?]B')
        self.assertNotIn('[?]', result)

    def test_unknown_at_end(self):
        result = self.corrector.correct('HELL[?]')
        self.assertNotIn('[?]', result)

    # ── token correctness (best-effort) ───────────────────────────────────────

    def test_fills_the_correctly(self):
        result = self.corrector.correct('TH[?] QUICK BROWN FOX')
        self.assertEqual(result[2], 'E')

    def test_fills_hello_correctly(self):
        result = self.corrector.correct('HELL[?] WORLD')
        self.assertEqual(result[4], 'O')

    # ── output type ───────────────────────────────────────────────────────────

    def test_returns_string(self):
        for text in ['MORSE', 'CQ DE', 'SOS', '73', '[?]']:
            self.assertIsInstance(self.corrector.correct(text), str)


if __name__ == '__main__':
    unittest.main()
