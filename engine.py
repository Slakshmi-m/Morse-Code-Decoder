import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, welch
from constants import MORSE_MAP

class MorseEngine:
    """Processes audio signals and decodes them into Morse text."""
    
    def __init__(self, sample_rate: int, data: np.ndarray):
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = data[:, 0]
        
        self.sample_rate = sample_rate

        # 1. Auto-detect tone frequency (Ninja files vary between 400Hz and 800Hz)
        tone_freq = self._detect_peak_frequency(data)
        
        # 2. Apply Bandpass filter centered at the detected frequency
        filtered_data = self._bandpass_filter(data, tone_freq - 100, tone_freq + 100)
        
        # Safely Normalize filtered signal
        abs_data = np.abs(filtered_data)
        max_val = np.max(abs_data)
        if max_val > 0:
            self.data = abs_data / max_val
        else:
            self.data = abs_data

    def _detect_peak_frequency(self, data):
        """Finds the most prominent frequency in the audio signal."""
        f, psd = welch(data, self.sample_rate, nperseg=1024)
        # Focus on typical Morse frequencies (300Hz - 1200Hz)
        mask = (f > 300) & (f < 1200)
        peak_freq = f[mask][np.argmax(psd[mask])] if np.any(mask) else 700
        return peak_freq

    def _bandpass_filter(self, data, lowcut, highcut, order=5):
        """Removes noise by only allowing frequencies between lowcut and highcut."""
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        # Ensure frequencies are within valid range
        low, high = max(0.01, low), min(0.99, high)
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def _get_binary_signal(self, threshold_factor=0.5):
        """Converts raw audio amplitude into a binary 0/1 (off/on) signal."""
        window_size = int(self.sample_rate * 0.005)  # 5ms
        smooth_data = np.convolve(self.data, np.ones(window_size)/window_size, mode='same')
        peak = np.percentile(smooth_data, 95)
        floor = np.percentile(smooth_data, 5)
        threshold = floor + (peak - floor) * threshold_factor
        return (smooth_data > threshold).astype(int)

    def _get_durations(self, binary_signal):
        """Groups consecutive 1s and 0s to find the length of pulses and silences."""
        changes = np.diff(binary_signal)
        change_indices = np.where(changes != 0)[0]
        
        # Add start and end indices
        indices = np.concatenate(([0], change_indices + 1, [len(binary_signal)]))
        durations = np.diff(indices)
        states = [binary_signal[i] for i in indices[:-1]]
        
        return list(zip(states, durations))

    def decode(self):
        """The main decoding loop using adaptive timing."""
        binary_signal = self._get_binary_signal()
        raw_segments = self._get_durations(binary_signal)

        # --- Step 1: rough unit estimate for noise filtering ---
        all_on = [d for s, d in raw_segments if s == 1]
        if not all_on:
            return "[No Morse Signal Detected]"
        rough_unit = np.percentile(all_on, 25)

        # Drop any segment shorter than rough_unit/3 — these are noise artifacts,
        # not real dots or silences.
        min_dur = max(1, int(rough_unit / 3))
        segments = [(s, d) for s, d in raw_segments if d >= min_dur]

        on_durations  = [d for s, d in segments if s == 1]
        off_durations = [d for s, d in segments if s == 0]
        if not on_durations or not off_durations:
            return "[No Morse Signal Detected]"

        # --- Step 2: refined unit (dit length) ---
        # The median lands in the dit cluster for typical English text (~60% dots).
        # Taking the mean of everything below the median gives a stable dit estimate.
        median_on = np.median(on_durations)
        dits = [d for d in on_durations if d < median_on]
        unit = float(np.mean(dits)) if dits else median_on / 2.0

        # --- Step 3: space thresholds ---
        # Standard Morse ratios: intra-symbol=1 unit, letter-gap=3 units, word-gap=7 units.
        # Setting char at 2.5× puts us safely above intra-symbol gaps (measured ~1.0–1.4×)
        # and below letter gaps (measured ~2.8–3.5×).
        # Setting word at 5.0× sits between the highest letter gap (~3.5×) and
        # lowest word gap (~7×), correctly separating words for standard and
        # Farnsworth-spaced audio.
        char_space_threshold = unit * 2.5
        word_space_threshold = unit * 5.0

        # --- Step 4: decode ---
        decoded_text = []
        current_morse = ""

        for state, duration in segments:
            if state == 1:
                current_morse += "." if duration < unit * 2 else "-"
            else:
                if duration > word_space_threshold:
                    if current_morse:
                        decoded_text.append(MORSE_MAP.get(current_morse, "[?]"))
                        current_morse = ""
                    decoded_text.append(" ")
                elif duration > char_space_threshold:
                    if current_morse:
                        decoded_text.append(MORSE_MAP.get(current_morse, "[?]"))
                        current_morse = ""

        if current_morse:
            decoded_text.append(MORSE_MAP.get(current_morse, "[?]"))

        return "".join(decoded_text).strip()

def decode_wav(file_path: str) -> str:
    """Helper function to load and decode a WAV file."""
    try:
        fs, data = wavfile.read(file_path)
        engine = MorseEngine(fs, data)
        return engine.decode()
    except Exception as e:
        return f"File Error: {str(e)}"