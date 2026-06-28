import unittest

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

from src.model import (
    VOCAB, VOCAB_SIZE, BLANK_IDX,
    CHAR_TO_IDX, IDX_TO_CHAR, greedy_decode,
)


class TestVocab(unittest.TestCase):

    def test_vocab_size_is_vocab_plus_blank(self):
        self.assertEqual(VOCAB_SIZE, len(VOCAB) + 1)

    def test_blank_idx_is_last(self):
        self.assertEqual(BLANK_IDX, len(VOCAB))

    def test_char_to_idx_covers_full_vocab(self):
        self.assertEqual(len(CHAR_TO_IDX), len(VOCAB))
        for char in VOCAB:
            self.assertIn(char, CHAR_TO_IDX)

    def test_idx_to_char_is_inverse_of_char_to_idx(self):
        for char, idx in CHAR_TO_IDX.items():
            self.assertEqual(IDX_TO_CHAR[idx], char)

    def test_all_indices_unique(self):
        indices = list(CHAR_TO_IDX.values())
        self.assertEqual(len(indices), len(set(indices)))

    def test_blank_not_in_char_to_idx(self):
        # BLANK_IDX must not collide with any character index
        self.assertNotIn(BLANK_IDX, set(CHAR_TO_IDX.values()))

    def test_space_in_vocab(self):
        self.assertIn(' ', VOCAB)

    def test_letters_in_vocab(self):
        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.assertIn(ch, VOCAB)

    def test_digits_in_vocab(self):
        for ch in '0123456789':
            self.assertIn(ch, VOCAB)


@unittest.skipUnless(_TORCH, 'torch not installed')
class TestGreedyDecode(unittest.TestCase):

    def _one_hot_log_probs(self, indices, length):
        """Create log_probs where each time step is one-hot at the given index (cycling)."""
        lp = torch.full((length, VOCAB_SIZE), -1e9)
        for t in range(length):
            lp[t, indices[t % len(indices)]] = 0.0
        return lp

    def test_all_blanks_gives_empty_string(self):
        lp = torch.full((10, VOCAB_SIZE), -1e9)
        lp[:, BLANK_IDX] = 0.0
        self.assertEqual(greedy_decode(lp), '')

    def test_repeated_char_collapses_to_one(self):
        a_idx = CHAR_TO_IDX['A']
        lp = self._one_hot_log_probs([a_idx], 5)
        self.assertEqual(greedy_decode(lp), 'A')

    def test_char_then_blank_then_same_char_gives_two(self):
        a_idx = CHAR_TO_IDX['A']
        lp = self._one_hot_log_probs([a_idx, BLANK_IDX, a_idx], 3)
        self.assertEqual(greedy_decode(lp), 'AA')

    def test_sos_sequence(self):
        indices = [CHAR_TO_IDX[c] for c in 'SOS']
        lp = self._one_hot_log_probs(indices, 3)
        self.assertEqual(greedy_decode(lp), 'SOS')

    def test_output_is_string(self):
        lp = torch.randn(20, VOCAB_SIZE)
        result = greedy_decode(lp.log_softmax(-1))
        self.assertIsInstance(result, str)


@unittest.skipUnless(_TORCH, 'torch not installed')
class TestMorseDecoderModel(unittest.TestCase):

    def setUp(self):
        from src.model import MorseDecoder
        self.model = MorseDecoder(n_mels=64, hidden=64, layers=1)
        self.model.eval()

    def test_output_shape_batch_1(self):
        x = torch.zeros(1, 1, 64, 50)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[2], VOCAB_SIZE)

    def test_output_shape_batch_4(self):
        x = torch.zeros(4, 1, 64, 80)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[2], VOCAB_SIZE)

    def test_output_time_dim_matches_input(self):
        x = torch.zeros(1, 1, 64, 100)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape[1], 100)

    def test_output_is_log_softmax(self):
        x = torch.randn(1, 1, 64, 40)
        with torch.no_grad():
            out = self.model(x)
        # log_softmax outputs are ≤ 0; exp should sum to ~1 per time step
        probs = out.exp()
        sums  = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
