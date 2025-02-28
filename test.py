import threading
import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# File path
file_path = "C:\\Thesis\\data\\Chilled 30 second (LOFI music_vibes) [sdZ-yo6c4T8].mp3"

# song loading 
y, sr = sf.read(file_path)
if y.ndim > 1:
    y = np.mean(y, axis=1)  # Convert to mono if stereo
# Add this after loading the audio (near the top of the file, after loading 'y' and 'sr')
# Modify the BPM calculation near the top of the file
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
tempo = np.float64(tempo[0] if isinstance(tempo, np.ndarray) else tempo)  # Safely convert tempo to float

# Shared data for threading
stft_result = None
mfcc_result = None
cqt_result = None  # Add a variable for CQT result
computation_lock = threading.Lock()  # Lock for thread-safe operations

# STFT Function 
def compute_stft():
    global stft_result
    print("Computing STFT...")
    n_fft = 2048
    D = librosa.stft(y, n_fft=n_fft)
    with computation_lock:
        stft_result = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print("STFT computation complete.")

# MFCC function
def compute_mfcc():
    global mfcc_result
    print("Computing MFCC...")
    with computation_lock:
        mfcc_result = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print("MFCC computation complete.")

# CQT function
def compute_cqt():
    global cqt_result
    print("Computing CQT...")
    C = librosa.cqt(y, sr=sr)
    with computation_lock:
        cqt_result = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    print("CQT computation complete.")

# creating and starting threads
stft_thread = threading.Thread(target=compute_stft)
mfcc_thread = threading.Thread(target=compute_mfcc)
cqt_thread = threading.Thread(target=compute_cqt)  # Add a thread for CQT

stft_thread.start()
mfcc_thread.start()
cqt_thread.start()  # Start the CQT thread

stft_thread.join()
mfcc_thread.join()
cqt_thread.join()  # Join the CQT thread

print("STFT, MFCC, and CQT computations are complete.")
# time axis for audio duration
n_fft = 2048
hop_length = n_fft // 4  #Default hop length
times = librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=sr, hop_length=hop_length)

# frequency axis
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# finding continuous segments
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

# frequency mask
freq_mask = np.zeros_like(freqs, dtype=bool)
freq_mask[(freqs >= 425) & (freqs <= 440)] = True  # Around 432 Hz

# Volume mask
vol_mask = np.logical_and(stft_result >= -50, stft_result <= 30)

# find timestamps where the metrics are met 
freq_presence = np.any(stft_result[freq_mask, :], axis=0)
vol_presence = np.any(vol_mask, axis=0)
combined_presence = freq_presence & vol_presence

# segmenting to ensure the algo doesnt analyze beyond the audio duration
max_time = librosa.get_duration(y=y, sr=sr)
combined_segments = find_segments(combined_presence, times)
filtered_segments = [(start, end) for start, end in combined_segments if end <= max_time]
# Create unified spectrogram before plotting
def create_unified_spectrogram():
    # Find the minimum number of frames among all spectrograms
    min_frames = min(stft_result.shape[1], mfcc_result.shape[1], cqt_result.shape[1])
    
    # Find maximum frequency index (height) for visualization
    max_freq_idx = stft_result.shape[0]
    
    # Initialize the RGB spectrogram array
    unified = np.zeros((max_freq_idx, min_frames, 3))
    
    # Normalize and add each component to the RGB channels
    unified[:, :min_frames, 0] = librosa.util.normalize(stft_result[:, :min_frames])  # Red channel (STFT)
    
    # Resize MFCC to match STFT dimensions
    mfcc_resized = librosa.util.normalize(np.resize(mfcc_result[:, :min_frames], (max_freq_idx, min_frames)))
    unified[:, :min_frames, 1] = mfcc_resized  # Green channel (MFCC)
    
    # Resize CQT to match STFT dimensions
    cqt_resized = librosa.util.normalize(np.resize(cqt_result[:, :min_frames], (max_freq_idx, min_frames)))
    unified[:, :min_frames, 2] = cqt_resized  # Blue channel (CQT)
    
    return unified, max_freq_idx, min_frames

# Create the unified spectrogram
unified_spectrogram, max_freq_idx, min_frames = create_unified_spectrogram()

# figure 
plt.figure(figsize=(15, 10))  # Adjust figure size to accommodate the unified plot

# Unified Spectrogram
plt.subplot(211)
plt.imshow(unified_spectrogram, aspect='auto', origin='lower',
           extent=[0, times[min_frames-1], 0, freqs[max_freq_idx-1]])

# Add a title that explains the color channels
plt.title('Unified Spectrogram (Red: STFT, Green: MFCC, Blue: CQT)', fontsize=14)

# Add colorbar
plt.colorbar(label='Normalized Intensity')

# Add frequency labels on y-axis
plt.ylabel('Frequency (Hz)')

# Reference lines
plt.axhline(y=432, color='yellow', linestyle='--', alpha=0.7)
plt.text(times[min_frames-1] + 0.5, 432, '432 Hz', color='yellow', fontweight='bold')

# Highlight sleep-conducive regions
for i, (start, end) in enumerate(filtered_segments, 1):
    plt.axvspan(start, end, color='cyan', alpha=0.3, edgecolor='cyan', linewidth=2)
    mid_point = (start + end) / 2
    plt.text(mid_point, freqs[max_freq_idx-1] * 0.9, f'Region {i}',
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='cyan'))

# Add annotations explaining what each color represents
plt.annotate('Red intensity: Frequency content (STFT)', 
             xy=(0.02, 0.98), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.3))
             
plt.annotate('Green intensity: Timbre information (MFCC)', 
             xy=(0.02, 0.93), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.3))
             
plt.annotate('Blue intensity: Harmonic structure (CQT)', 
             xy=(0.02, 0.88), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="blue", alpha=0.3))

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.show()
"""
# MFCC Display
plt.subplot(212)
librosa.display.specshow(mfcc_result, x_axis="time", sr=sr, cmap='coolwarm')
cbar2 = plt.colorbar()
cbar2.set_label('MFCC Magnitude')

# Highlight regions in MFCC plot
for i, (start, end) in enumerate(filtered_segments, 1):
    plt.axvspan(start, end, color='green', alpha=0.2)
    
    # Add timestamp text
    mid_point = (start + end) / 2
    plt.text(mid_point, mfcc_result.shape[0] - 1, f'Region {i}\n{start:.1f}s - {end:.1f}s',
             horizontalalignment='center',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.title("MFCCs with Sleep-Conducive Regions")"""

#kulang pa ng tresholding 

# Adjust layout
plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05, right=0.95, left=0.1)
plt.show()

# Remove the duplicate detailed analysis section and keep only one instance
print("\nDetailed Analysis of Sleep-Conducive Regions:")
for i, (start, end) in enumerate(filtered_segments, 1):
    duration = end - start
    time_start_idx = max(0, min(int(start * sr / hop_length), stft_result.shape[1] - 1))
    time_end_idx = max(0, min(int(end * sr / hop_length), stft_result.shape[1] - 1))

    if time_start_idx < time_end_idx:
        # Check if the slice is not empty before calculating the mean
        if stft_result[:, time_start_idx:time_end_idx].size > 0:
            avg_db = np.nanmean(stft_result[:, time_start_idx:time_end_idx])
        else:
            avg_db = float("nan")
        
        # Calculate local BPM for this segment
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
    
    # Calculate frequency distribution
    time_slice = librosa.stft(y, n_fft=min(n_fft, len(y)))[:, time_start_idx:time_end_idx]
    if time_slice.size > 0:
        freq_energy = np.mean(np.abs(time_slice), axis=1)
        peak_freq_idx = np.argmax(freq_energy)
        print(f"Peak frequency: {freqs[peak_freq_idx]:.1f} Hz")
    else:
        print("Peak frequency: nan Hz")
# Overall statistics (fix the order and remove duplicate)
# Overall statistics (keep only one instance)
total_duration = max_time
optimal_duration = sum(end - start for start, end in filtered_segments)
percentage_optimal = (optimal_duration / total_duration) * 100

# Single print section for statistics
print(f"\nOverall Statistics:")
print(f"Total audio duration: {total_duration:.1f}s")
print(f"Total duration of sleep-conducive segments: {optimal_duration:.1f}s")
print(f"Percentage of audio with optimal characteristics: {percentage_optimal:.1f}%")
print(f"Average BPM of entire track: {tempo:.1f}")
# Analysis of CQT and other features
def analyze_cqt():
    print("\nAnalyzing CQT and extracting features...")

    # Tonal & Harmonic Features
    chroma = librosa.feature.chroma_cqt(C=cqt_result, sr=sr)
    print("Chroma features calculated.")
    print(f"Chroma Shape: {chroma.shape}")
    print(f"Chroma Mean: {np.mean(chroma, axis=1)}")

    # Pitch & Melody
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    print("Pitch and melody features calculated.")
    pitch_mean = np.mean(pitches, axis=1)
    print(f"Pitch Mean: {pitch_mean}")

    # Frequency & Timbre
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    print("Frequency and timbre features calculated.")
    print(f"Spectral Centroid Mean: {np.mean(spectral_centroid):.2f}")
    print(f"Spectral Bandwidth Mean: {np.mean(spectral_bandwidth):.2f}")

    # Rhythm and Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = np.float64(tempo[0] if isinstance(tempo, np.ndarray) else tempo)  # Ensure tempo is a float
    print(f"Global Tempo: {tempo:.1f} BPM")

    # Peak Frequency Detection
    peak_freqs = np.argmax(cqt_result, axis=0)
    peak_freqs_hz = librosa.cqt_frequencies(cqt_result.shape[0], fmin=librosa.note_to_hz('C1'))[peak_freqs]
    print("Peak frequency detection complete.")
    print(f"Peak Frequencies (Hz): {peak_freqs_hz}")

    # Spectral Energy Distribution
    spectral_energy = np.sum(cqt_result, axis=0)
    print("Spectral energy distribution calculated.")
    print(f"Spectral Energy Mean: {np.mean(spectral_energy):.2f}")

    # Low-Frequency Emphasis
    low_freq_emphasis = np.mean(cqt_result[:int(cqt_result.shape[0] * 0.2), :], axis=0)
    print("Low-frequency emphasis calculated.")
    print(f"Low-Frequency Emphasis Mean: {np.mean(low_freq_emphasis):.2f}")
# Call the analysis function after all computations
analyze_cqt()
