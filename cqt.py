import threading
import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.colors as colors
from scipy.interpolate import RegularGridInterpolator

# File path
file_path = "C:\\Thesis\\data\\Indie\\Daniel Caesar, Rex Orange County - Rearrange My World (Audio).mp3"

# Song loading 
y, sr = sf.read(file_path)
if y.ndim > 1:
    y = np.mean(y, axis=1)  # Convert to mono if stereo

# BPM calculation
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
tempo = np.float64(tempo[0] if isinstance(tempo, np.ndarray) else tempo)  # Safely convert tempo to float

# Shared data for threading
stft_result = None
mfcc_result = None
cqt_result = None
spectral_contrast = None
spectral_centroid = None
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
        mfcc_result = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Increased from 13 to 20 for more detail
    print("MFCC computation complete.")

# CQT function
def compute_cqt():
    global cqt_result
    print("Computing CQT...")
    C = librosa.cqt(y, sr=sr)
    with computation_lock:
        cqt_result = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    print("CQT computation complete.")

# Additional features function
def compute_additional_features():
    global spectral_contrast, spectral_centroid
    print("Computing additional features...")
    with computation_lock:
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    print("Additional features computation complete.")

# Creating and starting threads
stft_thread = threading.Thread(target=compute_stft)
mfcc_thread = threading.Thread(target=compute_mfcc)
cqt_thread = threading.Thread(target=compute_cqt)
additional_features_thread = threading.Thread(target=compute_additional_features)

stft_thread.start()
mfcc_thread.start()
cqt_thread.start()
additional_features_thread.start()

stft_thread.join()
mfcc_thread.join()
cqt_thread.join()
additional_features_thread.join()

print("All feature computations are complete.")

# Time and frequency axes
n_fft = 2048
hop_length = n_fft // 4  # Default hop length
times = librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=sr, hop_length=hop_length)
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# Finding continuous segments
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

# Function to create unified feature representation
def create_unified_spectrogram():
    # Ensure all features have same time dimension
    min_frames = min(stft_result.shape[1], mfcc_result.shape[1], cqt_result.shape[1])
    
    # Prepare STFT component (frequency domain information)
    # Use only a portion of the STFT for better visualization (focus on relevant frequencies)
    max_freq_idx = np.searchsorted(freqs, 8000)  # Limit to 8kHz for better visibility
    stft_component = stft_result[:max_freq_idx, :min_frames]
    
    # Normalize STFT to 0-1 range for fusion
    stft_norm = (stft_component - np.min(stft_component)) / (np.max(stft_component) - np.min(stft_component))
    
    # Prepare MFCC component (timbre information)
    # Resize MFCC to match STFT frequency bins using interpolation
    mfcc_x = np.linspace(0, 1, mfcc_result.shape[1])
    mfcc_y = np.linspace(0, 1, mfcc_result.shape[0])
    
    mfcc_interp = RegularGridInterpolator((mfcc_y, mfcc_x), mfcc_result)
    
    mfcc_x_new = np.linspace(0, 1, min_frames)
    mfcc_y_new = np.linspace(0, 1, max_freq_idx)
    
    mfcc_component = mfcc_interp(np.meshgrid(mfcc_y_new, mfcc_x_new, indexing='ij'))
    
    # Normalize MFCC
    mfcc_norm = (mfcc_component - np.min(mfcc_component)) / (np.max(mfcc_component) - np.min(mfcc_component))
    
    # Prepare CQT component (harmonic structure information)
    # Resize CQT to match STFT frequency bins
    cqt_x = np.linspace(0, 1, cqt_result.shape[1])
    cqt_y = np.linspace(0, 1, cqt_result.shape[0])
    
    cqt_interp = RegularGridInterpolator((cqt_y, cqt_x), cqt_result)
    
    cqt_x_new = np.linspace(0, 1, min_frames)
    cqt_y_new = np.linspace(0, 1, max_freq_idx)
    
    cqt_component = cqt_interp(np.meshgrid(cqt_y_new, cqt_x_new, indexing='ij'))
    
    # Normalize CQT
    cqt_norm = (cqt_component - np.min(cqt_component)) / (np.max(cqt_component) - np.min(cqt_component))
    
    # Combine the normalized components with different weights in RGB channels
    # Red channel: STFT (frequency information)
    # Green channel: MFCC (timbre information)
    # Blue channel: CQT (harmonic information)
    unified_rgb = np.zeros((max_freq_idx, min_frames, 3))
    unified_rgb[:, :, 0] = stft_norm * 0.8  # Red: STFT
    unified_rgb[:, :, 1] = mfcc_norm * 0.9  # Green: MFCC
    unified_rgb[:, :, 2] = cqt_norm        # Blue: CQT
    
    return unified_rgb, max_freq_idx, min_frames

# FEATURE INTEGRATION - Create a unified sleep score
def calculate_unified_sleep_score():
    # Make sure all features have the same number of time frames
    min_frames = min(stft_result.shape[1], mfcc_result.shape[1], cqt_result.shape[1], 
                     spectral_contrast.shape[1], len(spectral_centroid))
    
    # Create empty array to hold score for each time frame
    unified_score = np.zeros(min_frames)
    
    # 1. STFT-based frequency and volume criteria - IMPROVED FREQUENCY DETECTION
    freq_mask = np.zeros_like(freqs, dtype=bool)
    # More precise detection of 432 Hz with narrower band
    freq_mask[(freqs >= 430) & (freqs <= 434)] = True  # Narrower band around 432 Hz
    freq_mask[(freqs >= 1) & (freqs <= 100)] = True    # Low frequencies for relaxation
    
    # Check for actual energy in these frequency bands, not just presence
    freq_energy_432 = np.mean(stft_result[(freqs >= 430) & (freqs <= 434), :min_frames], axis=0)
    freq_energy_low = np.mean(stft_result[(freqs >= 1) & (freqs <= 100), :min_frames], axis=0)
    
    # Normalize these energies
    freq_energy_432_norm = MinMaxScaler().fit_transform(freq_energy_432.reshape(-1, 1)).flatten()
    freq_energy_low_norm = MinMaxScaler().fit_transform(freq_energy_low.reshape(-1, 1)).flatten()
    
    # Combine with volume criteria
    vol_mask = np.logical_and(stft_result >= -50, stft_result <= 30)
    vol_presence = np.any(vol_mask[:, :min_frames], axis=0)
    
    # New STFT score combines actual energy levels with volume criteria
    stft_score = (freq_energy_432_norm * 0.3 + freq_energy_low_norm * 0.7) * vol_presence.astype(float)
    
    # 2. MFCC-based criteria - FIXED NORMALIZATION
    # Focus on first few MFCCs which relate to overall timbre
    mfcc_first_coefs = mfcc_result[:5, :min_frames]
    mfcc_variance = np.var(mfcc_first_coefs, axis=0)
    
    # Clip extreme values before normalization to prevent outliers from skewing results
    mfcc_variance_clipped = np.clip(mfcc_variance, np.percentile(mfcc_variance, 5), np.percentile(mfcc_variance, 95))
    
    # Normalize variance (lower is better for sleep - consistent sounds)
    mfcc_variance_norm = 1 - MinMaxScaler().fit_transform(mfcc_variance_clipped.reshape(-1, 1)).flatten()
    
    # Check for low energy in higher MFCCs (indicating smoother sounds)
    higher_mfcc_energy = np.mean(np.abs(mfcc_result[10:, :min_frames]), axis=0)
    higher_mfcc_score = 1 - MinMaxScaler().fit_transform(higher_mfcc_energy.reshape(-1, 1)).flatten()
    
    # 3. CQT-based criteria (harmonic richness and low-frequency emphasis)
    # Low frequency emphasis (lower 20% of CQT bins)
    low_freq_idx = int(cqt_result.shape[0] * 0.2)
    low_freq_energy = np.mean(cqt_result[:low_freq_idx, :min_frames], axis=0)
    
    # Normalize (higher is better for sleep - more bass content)
    low_freq_score = MinMaxScaler().fit_transform(low_freq_energy.reshape(-1, 1)).flatten()
    
    # Harmonic stability (less variation in harmonic content)
    chroma = librosa.feature.chroma_cqt(C=librosa.db_to_amplitude(cqt_result[:, :min_frames]), sr=sr)
    chroma_var = np.var(chroma, axis=0)
    chroma_stability = 1 - MinMaxScaler().fit_transform(chroma_var.reshape(-1, 1)).flatten()
    
    # 4. Spectral Centroid (lower is better for sleep - less high frequency content)
    spec_cent_norm = 1 - MinMaxScaler().fit_transform(
        spectral_centroid[:min_frames].reshape(-1, 1)).flatten()
    
    # 5. Spectral Contrast (lower contrast is better for sleep - smoother sound)
    contrast_mean = np.mean(spectral_contrast[:, :min_frames], axis=0)
    contrast_norm = 1 - MinMaxScaler().fit_transform(contrast_mean.reshape(-1, 1)).flatten()
    
    # 6. NEW: BPM-based score (slower is better for sleep)
    # Create a tempo score based on local tempo estimation
    frame_duration = hop_length / sr  # Duration of each frame in seconds
    window_size = int(3 / frame_duration)  # ~3 second windows for local tempo
    
    tempo_scores = np.zeros(min_frames)
    for i in range(0, min_frames, window_size//2):  # 50% overlap between windows
        end_idx = min(i + window_size, min_frames)
        if end_idx - i > window_size // 2:  # Only process if window is large enough
            frame_start = int(i * hop_length)
            frame_end = int(end_idx * hop_length)
            if frame_end - frame_start > sr:  # Ensure at least 1 second of audio
                segment = y[frame_start:frame_end]
                local_tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
                # Convert tempo to score (lower tempo = higher score)
                # Ideal sleep tempo is below 80 BPM
                tempo_score = max(0, min(1, 2 - local_tempo/80))
                tempo_scores[i:end_idx] = tempo_score
    
    # Fill any remaining zeros with the average
    if np.any(tempo_scores == 0) and np.any(tempo_scores > 0):
        tempo_scores[tempo_scores == 0] = np.mean(tempo_scores[tempo_scores > 0])
    
    # UPDATED WEIGHTS - Increased importance for BPM and timbre stability
    weights = {
        'stft': 0.10,              # Reduced from 0.15
        'mfcc_variance': 0.20,     # Increased from 0.15
        'mfcc_high_energy': 0.10,  # Reduced from 0.15
        'low_freq': 0.15,          # Reduced from 0.20
        'chroma_stability': 0.10,  # Reduced from 0.15
        'spectral_centroid': 0.05, # Reduced from 0.10
        'spectral_contrast': 0.05, # Reduced from 0.10
        'tempo': 0.25              # New weight for tempo
    }
    
    unified_score = (
        weights['stft'] * stft_score + 
        weights['mfcc_variance'] * mfcc_variance_norm + 
        weights['mfcc_high_energy'] * higher_mfcc_score +
        weights['low_freq'] * low_freq_score +
        weights['chroma_stability'] * chroma_stability +
        weights['spectral_centroid'] * spec_cent_norm +
        weights['spectral_contrast'] * contrast_norm +
        weights['tempo'] * tempo_scores
    )
    
    # Apply smoothing to prevent rapid fluctuations
    unified_score = np.convolve(unified_score, np.ones(5)/5, mode='same')
    
    return unified_score, {
        'stft_score': stft_score,
        'mfcc_variance': mfcc_variance_norm,
        'mfcc_high_energy': higher_mfcc_score,
        'low_freq': low_freq_score,
        'chroma_stability': chroma_stability,
        'spectral_centroid': spec_cent_norm,
        'spectral_contrast': contrast_norm,
        'tempo_score': tempo_scores
    }

# Create unified spectrogram
print("Creating unified spectrogram...")
unified_spectrogram, max_freq_idx, min_frames = create_unified_spectrogram()

# Calculate the unified sleep score
print("Calculating unified sleep score...")
sleep_score, component_scores = calculate_unified_sleep_score()  # Now returns component scores too

# Set threshold for sleep-conducive regions
threshold = 0.6  # Adjust based on validation
sleep_conducive_mask = sleep_score > threshold

# Find sleep-conducive segments
max_time = librosa.get_duration(y=y, sr=sr)
sleep_segments = find_segments(sleep_conducive_mask, times[:min_frames])
filtered_segments = [(start, end) for start, end in sleep_segments if end <= max_time]

# Create a single, comprehensive figure
plt.figure(figsize=(15, 10))

# Main plot: Unified RGB Spectrogram (uses all three spectral representations)
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

# Second plot: Sleep Score (derived from all features)
plt.subplot(212)
plt.plot(times[:min_frames], sleep_score, 'g-', linewidth=2)
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
plt.fill_between(times[:min_frames], 0, sleep_score, 
                 where=sleep_score > threshold, color='cyan', alpha=0.4)
plt.xlabel('Time (s)')
plt.ylabel('Sleep Score')
plt.title('Unified Sleep Score (Combined STFT, MFCC, and CQT Features)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Highlight sleep-conducive regions in the score plot too
for i, (start, end) in enumerate(filtered_segments, 1):
    plt.axvspan(start, end, color='cyan', alpha=0.3, edgecolor='cyan', linewidth=1)

# Add score component annotations
plt.annotate('Score components: Target frequencies (432Hz), Low-frequency presence, Timbre stability, \nHarmonic consistency, Low spectral centroid, and Smooth spectral transitions',
             xy=(0.02, 0.05), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.show()

# Detailed analysis of sleep-conducive regions with multi-feature information
print("\nDetailed Analysis of Sleep-Conducive Regions:")
for i, (start, end) in enumerate(filtered_segments, 1):
    duration = end - start
    
    # Find indices for time slice
    time_start_idx = max(0, min(int(start * sr / hop_length), stft_result.shape[1] - 1))
    time_end_idx = max(0, min(int(end * sr / hop_length), stft_result.shape[1] - 1))
    
    if time_start_idx < time_end_idx:
        # Calculate average scores for this region across all features
        region_idx = slice(time_start_idx, time_end_idx)
        
        # STFT features
        if stft_result[:, region_idx].size > 0:
            avg_db = np.nanmean(stft_result[:, region_idx])
        else:
            avg_db = float("nan")
            
        # Calculate local BPM for this segment
        segment_samples = y[int(start * sr):int(end * sr)]
        if len(segment_samples) > 0:
            local_tempo, _ = librosa.beat.beat_track(y=segment_samples, sr=sr)
            local_tempo = np.float64(local_tempo[0] if isinstance(local_tempo, np.ndarray) else local_tempo)
        else:
            local_tempo = float("nan")
            
        # MFCC features
        if time_end_idx <= mfcc_result.shape[1]:
            mfcc_slice = mfcc_result[:, time_start_idx:time_end_idx]
            mfcc_mean = np.mean(mfcc_slice, axis=1)
            mfcc_var = np.var(mfcc_slice, axis=1)
            mfcc_low_energy = np.mean(mfcc_slice[:5, :])  # Lower coefficients (timbre)
            mfcc_high_energy = np.mean(mfcc_slice[10:, :])  # Higher coefficients (details)
        else:
            mfcc_low_energy = mfcc_high_energy = float("nan")
            
        # CQT features
        if time_end_idx <= cqt_result.shape[1]:
            cqt_slice = cqt_result[:, time_start_idx:time_end_idx]
            low_freq_idx = int(cqt_slice.shape[0] * 0.2)
            low_freq_energy = np.mean(cqt_slice[:low_freq_idx, :])
            
            # Peak frequency using CQT for better pitch accuracy
            peak_freq_idx = np.argmax(np.mean(cqt_slice, axis=1))
            cqt_freqs = librosa.cqt_frequencies(cqt_result.shape[0], fmin=librosa.note_to_hz('C1'))
            peak_freq_cqt = cqt_freqs[peak_freq_idx] if peak_freq_idx < len(cqt_freqs) else float("nan")
            
            # Harmonic content
            if time_end_idx <= cqt_result.shape[1]:
                chroma = librosa.feature.chroma_cqt(C=librosa.db_to_amplitude(cqt_result[:, time_start_idx:time_end_idx]), sr=sr)
                chroma_var = np.var(chroma, axis=1)
                harmonic_stability = 1 - np.mean(chroma_var)
            else:
                harmonic_stability = float("nan")
        else:
            low_freq_energy = peak_freq_cqt = harmonic_stability = float("nan")
            
        # Calculate average sleep score for this region
        if len(sleep_score) > time_end_idx:
            region_score = np.mean(sleep_score[time_start_idx:time_end_idx])
        else:
            region_score = float("nan")
            
        # Unified feature importance analysis
        feature_importance = {}
        if time_end_idx <= min(stft_result.shape[1], mfcc_result.shape[1], cqt_result.shape[1]):
            # Extract region-specific features and normalize them
            freq_mask = np.zeros_like(freqs, dtype=bool)
            freq_mask[(freqs >= 425) & (freqs <= 440)] = True  # Around 432 Hz
            freq_mask[(freqs >= 1) & (freqs <= 100)] = True    # Low frequencies
            
            freq_presence_score = np.mean(np.any(stft_result[freq_mask, time_start_idx:time_end_idx], axis=0).astype(float))
            mfcc_var_score = 1 - np.mean(np.var(mfcc_result[:5, time_start_idx:time_end_idx], axis=0))
            low_freq_score = np.mean(cqt_result[:low_freq_idx, time_start_idx:time_end_idx])
            
            # Store normalized scores
            feature_importance = {
                "432Hz_presence": freq_presence_score,
                "timbre_stability": mfcc_var_score,
                "low_freq_presence": low_freq_score,
                "harmonic_stability": harmonic_stability
            }
        
        # Print comprehensive analysis
        print(f"\nRegion {i}:")
        print(f"Timestamp: {start:.1f}s - {end:.1f}s")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Sleep Score: {region_score:.2f} (threshold: {threshold:.2f})")
        
        print("\nIntegrated Feature Analysis:")
        print(f"  Average intensity: {avg_db:.1f} dB")
        print(f"  Local BPM: {local_tempo:.1f}")
        print(f"  Timbre stability: {mfcc_var_score:.2f}")
        print(f"  Low-frequency content: {low_freq_score:.2f}")
        print(f"  Peak frequency: {peak_freq_cqt:.1f} Hz")
        print(f"  Harmonic stability: {harmonic_stability:.2f}")
        
        print("\nFeature Contribution to Sleep Score:")
        for feature, value in feature_importance.items():
            print(f"  {feature}: {value:.2f}")

# Overall statistics
total_duration = max_time
optimal_duration = sum(end - start for start, end in filtered_segments)
percentage_optimal = (optimal_duration / total_duration) * 100

print(f"\nOverall Statistics:")
print(f"Total audio duration: {total_duration:.1f}s")
print(f"Total duration of sleep-conducive segments: {optimal_duration:.1f}s")
print(f"Percentage of audio with optimal characteristics: {percentage_optimal:.1f}%")
print(f"Average BPM of entire track: {tempo:.1f}")

# Summary of sleep-conduciveness criteria used in analysis
print("\nSleep-Conduciveness Criteria Summary:")
print("1. Frequency: Presence of 432 Hz (healing frequency) and low frequencies (1-100 Hz)")
print("2. Timbre: Stable, consistent timbre with low MFCC variance")
print("3. Harmony: Stable harmonic structures with minimal variations")
print("4. Rhythm: Consistent tempo, preferably below 80 BPM")
print("5. Spectral Properties: Low spectral centroid and contrast (smoother sound)")

print("\nThe unified spectrogram combines:")
print("- STFT (red channel): Shows precise frequency and time information")
print("- MFCC (green channel): Represents timbre and perceptual characteristics")
print("- CQT (blue channel): Highlights harmonic structure and tonal properties")
print("\nBrighter regions in each color channel indicate stronger presence of those features.")

# Add this after the main visualization code, before the detailed analysis section
# VALIDATION: Add a sanity check for known non-sleep-conducive music
print("\nValidation Check:")
# Check if the file is likely a non-sleep-conducive genre (metal, hard rock, etc.)
filename = file_path.lower()
non_sleep_genres = ['metal', 'rock', 'slipknot', 'psychosocial']
is_likely_non_sleep = any(genre in filename for genre in non_sleep_genres)

# Check if BPM is too high for sleep
bpm_too_high = tempo > 90

# Check if there's high spectral contrast (typical of energetic music)
avg_contrast = np.mean(component_scores['spectral_contrast'])
high_contrast = avg_contrast < 0.4  # Low score means high contrast

# Combine validation checks
validation_flags = []
if is_likely_non_sleep:
    validation_flags.append("Genre suggests non-sleep music")
if bpm_too_high:
    validation_flags.append(f"BPM ({tempo:.1f}) exceeds sleep threshold (90)")
if high_contrast:
    validation_flags.append("High spectral contrast suggests energetic music")

if validation_flags:
    print("WARNING: Potential false positive in sleep conduciveness detection:")
    for flag in validation_flags:
        print(f"- {flag}")
    print(f"Consider adjusting threshold (currently {threshold}) or reviewing algorithm weights.")
else:
    print("No validation warnings detected.")