# Identifying Sleep-Conducive Audio Features in Music Using Spectral Analysis, and Deep Learning with Prescriptive Analysis for Enhanced Sleep Quality

A Python-based tool for analyzing audio files and generating spectrograms with specific frequency detection capabilities.

## Features
- STFT (Short-time Fourier Transform) analysis
- MFCC (Mel-frequency cepstral coefficients) computation
- CQT (Constant-Q Transform) analysis
- Multi-threaded audio processing
- Frequency band detection (2-12 Hz and 425-440 Hz)
- Automated spectrogram generation and saving

## Requirements
- Python 3.x
- soundfile
- librosa
- numpy
- matplotlib

## Setup
1. Clone the repository
2. Install dependencies:
```pip install -r requirements.txt```

## Usage
1. Place your audio file in the `data` directory
2. Update the file path in `Spectogram.py`
3. Run the script:
```python Spectogram.py```

## Output
- Generates spectrograms in the `spectrograms` directory
- Provides analysis of optimal audio characteristics
