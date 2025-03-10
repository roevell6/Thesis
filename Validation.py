import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

file_path = "C:\\Thesis\\data\\Indie\\Daniel Caesar, Rex Orange County - Rearrange My World (Audio).mp3"

try:
    y, sr = sf.read(file_path, always_2d=True)  
    if y.ndim > 1:
        y = np.mean(y, axis=1) 

    # Normalize the audio signal to maximize volume
    y = (y / np.max(np.abs(y))).astype(np.float32)
except RuntimeError as e:
    print(f"Error loading audio file: {e}")
    exit(1)

# Compute the onset envelope
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# Estimate the tempo using the onset envelope
tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
tempo = np.float64(tempo[0] if isinstance(tempo, np.ndarray) else tempo)

# Initialize global results
stft_result = None
mfcc_result = None
cqt_result = None  

def compute_stft():
    global stft_result
    print("Computing STFT...")
    n_fft = min(2048, len(y))
    if n_fft % 2 != 0:  
        n_fft -= 1
    D = librosa.stft(y, n_fft=n_fft)
    stft_result = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print("STFT computation complete.")

def compute_mfcc():
    global mfcc_result
    print("Computing MFCC...")
    mfcc_result = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print("MFCC computation complete.")

def compute_cqt():
    global cqt_result
    print("Computing CQT...")
    try:
        C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'))
        cqt_result = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    except ValueError as e:
        print(f"Error in CQT computation: {e}")
        cqt_result = None
    print("CQT computation complete.")

# Run computations sequentially (no threading)
compute_stft()
compute_mfcc()
compute_cqt()

print("STFT, MFCC, and CQT computations are complete.")

# Time and frequency analysis
n_fft = 2048
hop_length = n_fft // 4 
times = librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=sr, hop_length=hop_length)
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

def find_segments(binary_array, times):
    segments = []
    start_idx = None
    min_duration = 1.0 
    for i in range(len(binary_array)):
        if binary_array[i] and start_idx is None:
            start_idx = i
        elif not binary_array[i] and start_idx is not None:
            if (times[i-1] - times[start_idx]) >= min_duration:
                segments.append((times[start_idx], times[i-1]))
            start_idx = None
    
    if start_idx is not None:
        if (times[-1] - times[start_idx]) >= min_duration:
            segments.append((times[start_idx], times[-1]))
    
    return segments

freq_mask = np.zeros_like(freqs, dtype=bool)
freq_mask[(freqs < 520)] = True  # All frequencies below 520 Hz are target frequencies

# Calculate average intensity of target frequencies relative to overall intensity
freq_intensities = np.mean(stft_result[freq_mask, :], axis=0)
overall_intensities = np.mean(stft_result, axis=0)
freq_presence = np.clip(freq_intensities / (overall_intensities + 1e-10), 0, 1)
freq_presence_percentage = np.mean(freq_presence) * 100

# Check if frequencies outside the target range are dominant
outside_freq_mask = ~freq_mask  # Frequencies 520 Hz and above
outside_freq_intensities = np.mean(stft_result[outside_freq_mask, :], axis=0)

# Calculate the relative intensity of non-target frequencies compared to target frequencies
# This ensures we're measuring dominance rather than just presence
target_avg_intensity = np.mean(freq_intensities)
outside_avg_intensity = np.mean(outside_freq_intensities)
outside_freq_percentage = min(100, (outside_avg_intensity / (target_avg_intensity + 1e-10)) * 100)

vol_mask = np.logical_and(stft_result >= -50, stft_result <= 30)

# Only consider segments where target frequencies are present and not dominated by outside frequencies
freq_presence = np.any(stft_result[freq_mask, :], axis=0)
vol_presence = np.any(vol_mask, axis=0)
combined_presence = freq_presence & vol_presence

max_time = librosa.get_duration(y=y, sr=sr)
combined_segments = find_segments(combined_presence, times)
filtered_segments = [(start, end) for start, end in combined_segments 
                    if end <= max_time and (end - start) >= 1.0]

print("\nDetailed Analysis of Sleep-Conducive Regions:")
for i, (start, end) in enumerate(filtered_segments, 1):
    duration = end - start
    time_start_idx = max(0, min(int(start * sr / hop_length), stft_result.shape[1] - 1))
    time_end_idx = max(0, min(int(end * sr / hop_length), stft_result.shape[1] - 1))

    if time_start_idx < time_end_idx:
        if stft_result[:, time_start_idx:time_end_idx].size > 0:
            avg_db = np.nanmean(stft_result[:, time_start_idx:time_end_idx])
        else:
            avg_db = float("nan")
        
        segment_samples = y[int(start * sr):int(end * sr)]
        if len(segment_samples) > 0:
            local_tempo, _ = librosa.beat.beat_track(y=segment_samples, sr=sr)
            local_tempo = np.float64(local_tempo[0] if isinstance(local_tempo, np.ndarray) else local_tempo)
        else:
            local_tempo = float("nan")
    else:
        avg_db = float("nan")
        local_tempo = float("nan")
    
    print(f"\nRegion {i}:")
    print(f"Timestamp: {start:.1f}s - {end:.1f}s")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Average intensity: {avg_db:.1f} dB")
    print(f"Local BPM: {local_tempo:.1f}")
    
    time_slice = librosa.stft(y, n_fft=min(n_fft, len(y)))[:, time_start_idx:time_end_idx]
    if time_slice.size > 0:
        freq_energy = np.mean(np.abs(time_slice), axis=1)
        peak_freq_idx = np.argmax(freq_energy)
        print(f"Peak frequency: {freqs[peak_freq_idx]:.1f} Hz")
    else:
        print("Peak frequency: nan Hz")

def analyze_sleep_conduciveness():
    total_duration = librosa.get_duration(y=y, sr=sr)
    optimal_duration = sum(end - start for start, end in filtered_segments)
    percentage_optimal = (optimal_duration / total_duration) * 100
    
    freq_presence_percentage = np.mean(freq_presence) * 100
    
    volume_consistency = np.mean(vol_presence) * 100
    
    is_tempo_conducive = 60 <= tempo <= 80
    
    # Penalize if frequencies outside the target range are too dominant
    freq_balance_factor = max(0, 1 - (outside_freq_percentage / 100))
    
    score = (
        0.4 * percentage_optimal *  freq_balance_factor +  
        0.25 * freq_presence_percentage *  freq_balance_factor +  
        0.25 * volume_consistency +  
        0.1 * (60 <= tempo <= 80) * 100  
    )
    
    print("\nSleep Conduciveness Analysis:")
    print(f"Overall Score: {score:.1f}/100")
    print(f"\nDetailed Metrics:")
    print(f"- Optimal characteristics coverage: {percentage_optimal:.1f}%")
    print(f"- Tempo: {tempo:.1f} BPM ")
    print(f"- Target frequency presence (below 520 Hz): {freq_presence_percentage:.1f}%")
    print(f"- Non-target frequency presence (520 Hz and above): {outside_freq_percentage:.1f}%")
    print(f"- Volume consistency: {volume_consistency:.1f}%")
    
    if score >= 80:
        print("\nVerdict: Highly sleep-conducive")
    elif score >= 60:
        print("\nVerdict: Moderately sleep-conducive")
    else:
        print("\nVerdict: Not particularly sleep-conducive")

if __name__ == "__main__":
    analyze_sleep_conduciveness()
