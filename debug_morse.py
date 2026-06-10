import numpy as np
import scipy.io.wavfile as wav

sample_rate = 8000
speed_wpm = 15
frequency = 700
text = """
THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. 
MORSE CODE NINJA TRAINING IS ESSENTIAL FOR 
DEVELOPING HIGH SPEED TELEGRAPHY SKILLS.
"""

morse_dict = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', ' ': ' ', '.': '.-.-.-', ',': '--..--',
    '?': '..--..', '/': '-..-.', '-': '-....-', '(': '-.--.', ')': '-.--.-'
}

# timings
text = text.replace('\n', ' ').upper()
text = "".join([c for c in text if c in morse_dict])

dot_duration = 1.2 / speed_wpm
dash_duration = dot_duration * 3
intra_symbol_duration = dot_duration
letter_space_duration = dot_duration * 3
word_space_duration = dot_duration * 7

audio_signal = []

def add_tone(duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t) * 16384
    audio_signal.extend(tone)


def add_silence(duration):
    silence = np.zeros(int(sample_rate * duration))
    audio_signal.extend(silence)

for i, char in enumerate(text):
    if char == ' ':
        add_silence(word_space_duration)
        continue
    symbols = morse_dict[char]
    for j, symbol in enumerate(symbols):
        if symbol == '.':
            add_tone(dot_duration)
        elif symbol == '-':
            add_tone(dash_duration)
        if j < len(symbols) - 1:
            add_silence(intra_symbol_duration)
    if i < len(text) - 1 and text[i+1] != ' ':
        add_silence(letter_space_duration)

# Convert to numpy array and add NOISE (to make it harder for the decoder)
signal_array = np.array(audio_signal)
noise = np.random.normal(0, 2000, signal_array.shape) # Add static noise
noisy_signal = (signal_array + noise).astype(np.int16)

wav.write('noisy_paragraph.wav', sample_rate, noisy_signal)
print('WAV written: noisy_paragraph.wav (with noise)')

sample_rate, data = wav.read('noisy_paragraph.wav')
print('read rate', sample_rate, 'shape', data.shape, 'dtype', data.dtype)
if data.ndim > 1:
    data = data[:, 0]
amp = np.abs(data)
threshold = 0.15 * np.max(amp)
print('threshold', threshold, 'max', np.max(amp))
binary_signal = (amp > threshold).astype(int)
print('binary ones', binary_signal.sum(), 'length', len(binary_signal))

current_state = binary_signal[0]
count = 0
tones = []
silences = []
for bit in binary_signal:
    if bit == current_state:
        count += 1
    else:
        dur = (count / sample_rate) * 1000
        if current_state == 1:
            tones.append(dur)
        else:
            silences.append(dur)
        current_state = bit
        count = 1
if count > 0:
    dur = (count / sample_rate) * 1000
    if current_state == 1:
        tones.append(dur)
    else:
        silences.append(dur)
print('tones len', len(tones), 'silences len', len(silences))
print('tones sample', tones[:20])
print('silences sample', silences[:20])

base_unit = np.percentile(tones, 15)
dot_dash_boundary = base_unit * 2
letter_word_boundary = base_unit * 5
print('base_unit', base_unit, 'dot_dash_boundary', dot_dash_boundary, 'letter_word_boundary', letter_word_boundary)

morse_string = ''
for i in range(len(tones)):
    morse_string += '.' if tones[i] < dot_dash_boundary else '-'
    if i < len(silences):
        if silences[i] >= letter_word_boundary:
            morse_string += '   '
        elif silences[i] >= dot_dash_boundary:
            morse_string += ' '

print('morse_string repr', repr(morse_string))
print('morse_string len', len(morse_string))

rev = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    "-----": "0", ".-.-.-": "."
}

decoded = []
for word in morse_string.strip().split('   '):
    decoded_letters = []
    for letter in word.split(' '):
        decoded_letters.append(rev.get(letter, '?'))
    decoded.append(''.join(decoded_letters))
print('decoded words', decoded)
print('final decoded', ' '.join(decoded))
