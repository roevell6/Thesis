import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import threading
from concurrent.futures import ThreadPoolExecutor

file_path = "C:\\Thesis\\data\\The Maras  Over The Moon.mp3"

y, sr = sf.read(file_path)
if y.ndim > 1:
    y = np.mean(y, axis=1)

class AudioProcessingResults:
    def __init__(self):
        self.D = None
        self.D_db = None
        self.mfccs = None
        self.C = None
        self.C_db = None

def compute_stft(y, n_fft, results):
    D = librosa.stft(y, n_fft=n_fft)
    results.D = D
    results.D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

def compute_mfccs(y, sr, results):
    results.mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

def compute_cqt(y, sr, results):
    C = librosa.cqt(y, sr=sr)
    results.C = C
    results.C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

y, sr = sf.read(file_path)
if y.ndim > 1:
    y = np.mean(y, axis=1)

results = AudioProcessingResults()
n_fft = 2048

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(compute_stft, y, n_fft, results),
        executor.submit(compute_mfccs, y, sr, results),
        executor.submit(compute_cqt, y, sr, results)
    ]
    for future in futures:
        future.result()

D = results.D
D_db = results.D_db
mfccs = results.mfccs
C = results.C
C_db = results.C_db

times = librosa.times_like(D)
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

def find_segments(binary_array, times):
    segments = []
    start_idx = None
    
    for i in range(len(binary_array)):
        if binary_array[i] and start_idx is None:
            start_idx = i
        elif not binary_array[i] and start_idx is not None:
            segments.append((times[start_idx], times[i-1]))
            start_idx = None
    
    if start_idx is not None:
        segments.append((times[start_idx], times[-1]))
    
    return segments

freq_mask = np.zeros_like(freqs, dtype=bool)
freq_mask[(freqs >= 2) & (freqs <= 12)] = True  
freq_mask[(freqs >= 425) & (freqs <= 440)] = True  

vol_mask = np.logical_and(D_db >= -50, D_db <= 30)

freq_presence = np.any(D_db[freq_mask, :], axis=0)
vol_presence = np.any(vol_mask, axis=0)
combined_presence = freq_presence & vol_presence

combined_segments = find_segments(combined_presence, times)




def get_next_spectrogram_number():
    base_path = 'c:\\Thesis\\spectrograms'
    existing_files = [f for f in os.listdir(base_path) if f.startswith('spectrogram_') and f.endswith('.png')]
    if not existing_files:
        return 1
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    return max(numbers) + 1

next_number = get_next_spectrogram_number()
save_path = f'c:\\Thesis\\spectrograms\\spectrogram_{next_number}.png'

plt.figure(figsize=(15, 6))
img = librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="log", cmap='coolwarm')
plt.axis('off')
plt.tight_layout(pad=0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()
plt.close()

total_duration = times[-1]
optimal_duration = sum(end - start for start, end in combined_segments)
percentage_optimal = (optimal_duration / total_duration) * 100

print(f"\nPercentage of audio with optimal characteristics: {percentage_optimal:.1f}%")