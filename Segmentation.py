import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import threading
from concurrent.futures import ThreadPoolExecutor

file_path = "C:\\Thesis\\data\\Apocalypse - Cigarettes After Sex [ ezmp3.cc ].mp3"

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

def get_next_sleep_conducive_number():
    base_path = 'c:\\Thesis\\spectrograms'
    existing_files = [f for f in os.listdir(base_path) if f.startswith('Sleep Conducive_') and f.endswith('.png')]
    if not existing_files:
        return 1
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    return max(numbers) + 1 if numbers else 1

def process_segment(y_segment, sr, start_time, segment_number):
    # Compute STFT for the segment
    D = librosa.stft(y_segment, n_fft=n_fft)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Get the next available number
    next_number = get_next_sleep_conducive_number()
    
    # Save spectrogram for the segment with specific dimensions
    plt.figure(figsize=(2.24, 2.24))  # Set figure size to match desired output
    librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="log", cmap='coolwarm')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Use the Sleep Conducive naming convention with incremental numbers
    save_path = f'c:\\Thesis\\spectrograms\\Sleep Conducive_{next_number}.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)  # Adjusted DPI to get 224x224
    plt.close()

# Segment the audio into 5-second chunks
segment_duration = 5  # seconds
samples_per_segment = int(sr * segment_duration)
total_samples = len(y)
num_segments = int(np.ceil(total_samples / samples_per_segment))

for i in range(num_segments):
    start_sample = i * samples_per_segment
    end_sample = min((i + 1) * samples_per_segment, total_samples)
    segment = y[start_sample:end_sample]
    start_time = i * segment_duration
    
    process_segment(segment, sr, start_time, i + 1)

# Calculate and print the percentage of optimal characteristics for the entire audio
total_duration = times[-1]
optimal_duration = sum(end - start for start, end in combined_segments)
percentage_optimal = (optimal_duration / total_duration) * 100

print(f"\nPercentage of audio with optimal characteristics: {percentage_optimal:.1f}%")