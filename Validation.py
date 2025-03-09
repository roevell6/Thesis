import threading
import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

file_path = "C:\\Thesis\\data\\Hip-Hop\\6LACK  Pretty Little Fears ft J Cole Official Music Video.mp3"

try:
    y, sr = sf.read(file_path, always_2d=True)  
    if y.ndim > 1:
        y = np.mean(y, axis=1) 

    # Normalize the audio signal to maximize volume
    y = y / np.max(np.abs(y))
except RuntimeError as e:
    print(f"Error loading audio file: {e}")
    exit(1)

# Compute the onset envelope
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# Estimate the tempo using the onset envelope
tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
tempo = np.float64(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
stft_result = None
mfcc_result = None
cqt_result = None  
computation_lock = threading.Lock()  

def compute_stft():
    global stft_result
    print("Computing STFT...")
    n_fft = min(2048, len(y))
    if n_fft % 2 != 0:  
        n_fft -= 1
    D = librosa.stft(y, n_fft=n_fft)
    with computation_lock:
        stft_result = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print("STFT computation complete.")

def compute_mfcc():
    global mfcc_result
    print("Computing MFCC...")
    with computation_lock:
        mfcc_result = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print("MFCC computation complete.")

def compute_cqt():
    global cqt_result
    print("Computing CQT...")
    try:
        C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'))
        with computation_lock:
            cqt_result = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    except ValueError as e:
        print(f"Error in CQT computation: {e}")
        cqt_result = None
    print("CQT computation complete.")

stft_thread = threading.Thread(target=compute_stft)
mfcc_thread = threading.Thread(target=compute_mfcc)
cqt_thread = threading.Thread(target=compute_cqt) 

stft_thread.start()
mfcc_thread.start()
cqt_thread.start() 

stft_thread.join()
mfcc_thread.join()
cqt_thread.join()  

print("STFT, MFCC, and CQT computations are complete.")
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
freq_mask[(freqs >= 425) & (freqs <= 440)] = True  

vol_mask = np.logical_and(stft_result >= -50, stft_result <= 30)

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
    
    # Use the global tempo variable instead of local_tempo
    is_tempo_conducive = 60 <= tempo <= 80
    score = (
        0.4 * percentage_optimal +  # Adjusted weight to 40%
        0.25 * freq_presence_percentage +  # Unchanged weight
        0.25 * volume_consistency +  # Unchanged weight
        0.1 * (60 <= tempo <= 80) * 100  # Added BPM analysis at 10%
    )
    
    print("\nSleep Conduciveness Analysis:")
    print(f"Overall Score: {score:.1f}/100")
    print(f"\nDetailed Metrics:")
    print(f"- Optimal characteristics coverage: {percentage_optimal:.1f}%")
    print(f"- Tempo: {tempo:.1f} BPM ")
    print(f"- Target frequency presence: {freq_presence_percentage:.1f}%")
    print(f"- Volume consistency: {volume_consistency:.1f}%")
    
    if score >= 80:
        print("\nVerdict: Highly sleep-conducive")
    elif score >= 60:
        print("\nVerdict: Moderately sleep-conducive")
    else:
        print("\nVerdict: Not particularly sleep-conducive")

if __name__ == "__main__":
    threads = [
        threading.Thread(target=compute_stft),
        threading.Thread(target=compute_mfcc),
        threading.Thread(target=compute_cqt)
    ]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    analyze_sleep_conduciveness()
