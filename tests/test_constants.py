import unittest
from src.constants import MORSE_MAP, TEXT_TO_MORSE


class TestMorseMap(unittest.TestCase):

    def test_known_entries(self):
        self.assertEqual(MORSE_MAP['.-'],    'A')
        self.assertEqual(MORSE_MAP['---'],   'O')
        self.assertEqual(MORSE_MAP['.....'], '5')
        self.assertEqual(MORSE_MAP['.-.-.-'], '.')
        self.assertEqual(MORSE_MAP['--..--'], ',')

    def test_all_26_letters_present(self):
        letters = {v for v in MORSE_MAP.values() if isinstance(v, str) and v.isalpha()}
        self.assertEqual(len(letters), 26)

    def test_all_10_digits_present(self):
        digits = {v for v in MORSE_MAP.values() if isinstance(v, str) and v.isdigit()}
        self.assertEqual(len(digits), 10)

    def test_patterns_contain_only_dots_and_dashes(self):
        for pattern in MORSE_MAP:
            if pattern != ' ':
                self.assertTrue(all(c in '.-' for c in pattern),
                                f"Unexpected char in pattern: {pattern!r}")

    def test_all_patterns_unique(self):
        patterns = [p for p in MORSE_MAP if p != ' ']
        self.assertEqual(len(patterns), len(set(patterns)))


class TestTextToMorse(unittest.TestCase):

    def test_is_inverse_of_morse_map(self):
        for pattern, char in MORSE_MAP.items():
            if char != ' ':
                self.assertEqual(TEXT_TO_MORSE[char], pattern,
                                 f"Round-trip failed for '{char}'")

    def test_known_encodings(self):
        self.assertEqual(TEXT_TO_MORSE['A'], '.-')
        self.assertEqual(TEXT_TO_MORSE['S'], '...')
        self.assertEqual(TEXT_TO_MORSE['O'], '---')
        self.assertEqual(TEXT_TO_MORSE['0'], '-----')

    def test_covers_same_characters_as_morse_map(self):
        # Exclude the space→space identity entry that constants.py carries
        chars_in_map = {v for v in MORSE_MAP.values() if v != ' '}
        chars_in_t2m = {k for k in TEXT_TO_MORSE.keys() if k != ' '}
        self.assertEqual(chars_in_map, chars_in_t2m)


if __name__ == '__main__':
    unittest.main()
