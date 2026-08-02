import numpy as np
import torch
import sys
sys.path.insert(0, '.')
import torchaudio.transforms as T
from ml.model import MorseDecoder, greedy_decode

MORSE = {
    'E': '.', 'A': '.-', 'S': '...', 'O': '---',
    'C': '-.-.', 'Q': '--.-', 'H': '....', 'M': '--',
    'B': '-...', 'K': '-.-', 'T': '-', 'N': '-.', 'X': '-..-',
}

def synth(text):
    dot, fs, freq = 1.2 / 22, 8000, 700.0
    segs = [np.zeros(int(fs * 0.3))]
    for ch in text:
        for sym in MORSE.get(ch, ''):
            t = np.arange(int((dot * 3 if sym == '-' else dot) * fs)) / fs
            segs.append((np.sin(2 * 3.14159 * freq * t) * 0.5).astype('float32'))
            segs.append(np.zeros(int(dot * fs)))
        segs.append(np.zeros(int(dot * 3 * fs)))
    segs.append(np.zeros(int(fs * 0.3)))
    return (np.concatenate(segs) * 16384).clip(-32767, 32767).astype('int16')

ckpt  = torch.load('model_best.pt', map_location='cpu', weights_only=False)
model = MorseDecoder(n_mels=ckpt.get('n_mels', 32)).eval()
model.load_state_dict(ckpt['model_state'])
print(f"Loaded: epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")

mel = T.MelSpectrogram(8000, n_fft=256, hop_length=64, n_mels=32)
db  = T.AmplitudeToDB()

words   = ['E', 'A', 'SOS', 'CQ', 'HAM', 'AB', 'OK', 'TNX']
correct = 0
print('\n--- Quick decode test ---')
for word in words:
    s    = synth(word)
    spec = db(mel(torch.FloatTensor(s.astype('float32') / 32768).unsqueeze(0))).unsqueeze(0)
    got  = greedy_decode(model(spec)[0]).strip()
    ok   = got == word
    correct += int(ok)
    tag  = '[OK]   ' if ok else '[WRONG]'
    print(f'  {tag}  expected={word:<6}  got={repr(got)}')

print(f'\nScore: {correct}/8')
if correct == 8:
    print('Perfect — model is ready!')
elif correct >= 5:
    print('Model looks good!')
else:
    print('Model still needs more training.')
