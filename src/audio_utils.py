"""
Step 3.1 — Audio Loading, Preprocessing & Acoustic Feature Extraction (`src/audio_utils.py`)

This module handles Step 1 & Step 2 of the VoiceScribe pipeline:
1. Decoding any audio/video file (.mp3, .wav, .mp4) via FFmpeg into a 16 kHz mono float32 waveform.
2. Computing basic signal statistics (duration, sample count, RMS energy, peak amplitude).
3. Extracting the 80-bin Log-Mel Spectrogram (input representation for OpenAI Whisper).
4. Extracting 13 Mel-Frequency Cepstral Coefficients (MFCCs) + fixed-length summary vectors
   (mean + std) for our custom English/French Scikit-Learn classifier.
5. Plotting Waveform vs. Log-Mel Spectrogram vs. MFCCs for EDA and presentation slides.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import whisper

# Standard sample rate required by OpenAI Whisper (16,000 samples per second)
SAMPLE_RATE: int = 16000


def load_and_preprocess_audio(
    file_path: Union[str, Path], target_sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Load an audio or video file (.wav, .mp3, .mp4, .m4a) using FFmpeg and standardize it
    to a 1D mono float32 NumPy array in the range [-1.0, 1.0] at `target_sr` (16 kHz).

    Parameters
    ----------
    file_path : str or Path
        Path to the input audio/video file.
    target_sr : int, default=16000
        Target sampling rate in Hz.

    Returns
    -------
    waveform : np.ndarray of shape (num_samples,), dtype=float32
        1D standardized mono audio waveform.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # whisper.load_audio calls FFmpeg under the hood to convert any media format
    # to 16 kHz, 1-channel (mono), 32-bit floating-point PCM samples.
    waveform = whisper.load_audio(str(path), sr=target_sr)
    return waveform


def get_audio_stats(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, float]:
    """
    Calculate basic signal statistics of a 1D waveform for dataset EDA and sanity checks.

    Math Notes for Presentation:
    - Duration (s) = N / sr
    - RMS Energy   = sqrt( (1/N) * sum(x[i]^2) )  -> measures average loudness
    - Peak Amp     = max(|x[i]|)                  -> maximum signal excursion
    """
    num_samples = int(waveform.shape[0])
    duration_sec = round(num_samples / float(sr), 3)
    rms_energy = float(np.sqrt(np.mean(np.square(waveform)))) if num_samples > 0 else 0.0
    peak_amp = float(np.max(np.abs(waveform))) if num_samples > 0 else 0.0

    return {
        "num_samples": num_samples,
        "sample_rate_hz": sr,
        "duration_sec": duration_sec,
        "rms_energy": round(rms_energy, 5),
        "peak_amplitude": round(peak_amp, 5),
    }


def extract_log_mel_spectrogram(
    waveform: np.ndarray, pad_to_30s: bool = False
) -> np.ndarray:
    """
    Convert a 16 kHz waveform into an 80-bin Log-Mel Spectrogram using Whisper's exact
    Short-Time Fourier Transform (STFT) parameters (25 ms window, 10 ms hop, 80 Mel bins).

    Parameters
    ----------
    waveform : np.ndarray of shape (num_samples,)
        16 kHz mono audio waveform.
    pad_to_30s : bool, default=False
        If True, pads/trims the audio to exactly 30 seconds (480,000 samples), producing
        shape (80, 3000) as required by Whisper's Transformer Encoder.
        If False, computes the spectrogram over the clip's actual duration (ideal for plots).

    Returns
    -------
    log_mel : np.ndarray of shape (80, n_frames)
        2D Log-Mel Spectrogram matrix.
    """
    audio_input = whisper.pad_or_trim(waveform) if pad_to_30s else waveform
    mel_tensor = whisper.log_mel_spectrogram(audio_input)
    return mel_tensor.cpu().numpy()


def extract_mfcc_features(
    waveform: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = 13
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs) using Librosa.

    Why Two Outputs?
    1. `mfcc_matrix` of shape (n_mfcc, n_frames): shows how vocal tract acoustics change
       frame-by-frame over time (used for visualization).
    2. `feature_vector` of shape (2 * n_mfcc,) = (26,): concatenates the temporal MEAN
       (13 values) and STANDARD DEVIATION (13 values) across frames so clips of different
       durations all map to a fixed 26-dimensional vector for Scikit-Learn classifiers.

    Returns
    -------
    mfcc_matrix : np.ndarray of shape (n_mfcc, n_frames)
    feature_vector : np.ndarray of shape (2 * n_mfcc,)
    """
    mfcc_matrix = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc_matrix, axis=1)
    mfcc_std = np.std(mfcc_matrix, axis=1)
    feature_vector = np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)
    return mfcc_matrix, feature_vector


def plot_audio_features(
    waveform: np.ndarray,
    log_mel: np.ndarray,
    mfcc_matrix: np.ndarray,
    title: str = "Audio Feature Inspection",
    sr: int = SAMPLE_RATE,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a 3-panel visualization showing:
      1. Raw 16 kHz Waveform (Time vs. Amplitude)
      2. 80-Bin Log-Mel Spectrogram (Whisper Encoder input)
      3. 13-Band MFCCs (Custom Language Classifier input)
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Panel 1: Time-domain waveform
    time_axis = np.linspace(0, len(waveform) / sr, num=len(waveform))
    axes[0].plot(time_axis, waveform, color="#2563eb", linewidth=0.7)
    axes[0].set_title("1. Standardized 16 kHz Mono Waveform (Time Domain)", fontsize=10.5)
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, max(time_axis[-1], 0.01))
    axes[0].grid(True, alpha=0.3)

    # Panel 2: 80-bin Log-Mel Spectrogram
    img_mel = axes[1].imshow(
        log_mel,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        extent=[0, len(waveform) / sr, 0, log_mel.shape[0]],
    )
    axes[1].set_title(
        "2. 80-Bin Log-Mel Spectrogram (What Whisper Transformer Reads)", fontsize=10.5
    )
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Mel Bin (0-80)")
    fig.colorbar(img_mel, ax=axes[1], format="%+2.1f")

    # Panel 3: 13-band MFCCs
    img_mfcc = librosa.display.specshow(
        mfcc_matrix, x_axis="time", sr=sr, ax=axes[2], cmap="coolwarm"
    )
    axes[2].set_title(
        "3. 13 Mel-Frequency Cepstral Coefficients — MFCCs (What Our Custom ML Classifier Reads)",
        fontsize=10.5,
    )
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("MFCC Index")
    fig.colorbar(img_mfcc, ax=axes[2], format="%+2.0f")

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)

    return fig
