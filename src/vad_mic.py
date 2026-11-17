r"""
Phase 4 — Voice Activity Detection (VAD) & Live Microphone Capture (`src/vad_mic.py`)

This module powers Live Microphone Transcription in VoiceScribe:
1. Energy-based Voice Activity Detection (`apply_vad_filter`):
   - Detects when a user is actually speaking vs. background room silence.
   - Prevents Whisper's classic failure mode: hallucinating phantom phrases
     (like "Sous-titres réalisés par..." or "Thank you for watching") on silent audio.
   - Splits or trims silence using frame-level decibel thresholds (`top_db`).
2. Direct Microphone Recording (`record_from_microphone`):
   - Captures 16 kHz mono audio directly from the system microphone via `sounddevice`.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

from src.audio_utils import SAMPLE_RATE


def apply_vad_filter(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    top_db: float = 28.0,
    min_rms_threshold: float = 0.008,
    min_speech_sec: float = 0.25,
    pad_sec: float = 0.10,
) -> Dict[str, Any]:
    """
    Apply Voice Activity Detection (VAD) to a 16 kHz mono waveform.

    Math & Algorithm (for Presentation):
    1. Global Silence Gate:
       Computes overall Root-Mean-Square (RMS) energy:
           RMS = sqrt( (1/N) * sum(x[i]^2) )
       If Peak < min_rms_threshold, the recording is pure silence/room noise -> `is_silent=True`.
    2. Frame-Level Decibel Segmentation:
       Uses `librosa.effects.split(waveform, top_db=top_db)` to find all audio intervals
       whose energy is within `top_db` decibels of the peak loudness.
    3. Concatenation & Padding:
       Adds a small `pad_sec` (100 ms) buffer around each voiced phrase so consonants at word
       boundaries are never clipped, and concatenates voiced segments.

    Returns
    -------
    dict with:
      - `cleaned_waveform`    : np.ndarray (silence-trimmed 16 kHz waveform)
      - `voiced_intervals_sec`: list of (start_sec, end_sec) tuples
      - `raw_duration_sec`    : float
      - `voiced_duration_sec` : float
      - `silence_removed_sec` : float
      - `speech_ratio`        : float in [0.0, 1.0]
      - `is_silent`           : bool (True if no human speech was detected)
    """
    raw_len = int(len(waveform))
    raw_duration_sec = raw_len / float(sr) if raw_len > 0 else 0.0

    if raw_len == 0:
        return {
            "cleaned_waveform": waveform,
            "voiced_intervals_sec": [],
            "raw_duration_sec": 0.0,
            "voiced_duration_sec": 0.0,
            "silence_removed_sec": 0.0,
            "speech_ratio": 0.0,
            "is_silent": True,
        }

    rms_energy = float(np.sqrt(np.mean(np.square(waveform))))
    peak_amp = float(np.max(np.abs(waveform)))

    # Gate 1: Check if recording is just room noise / dead air
    if rms_energy < (min_rms_threshold * 0.4) or peak_amp < min_rms_threshold:
        return {
            "cleaned_waveform": np.array([], dtype=np.float32),
            "voiced_intervals_sec": [],
            "raw_duration_sec": round(raw_duration_sec, 3),
            "voiced_duration_sec": 0.0,
            "silence_removed_sec": round(raw_duration_sec, 3),
            "speech_ratio": 0.0,
            "is_silent": True,
        }

    # Gate 2: Split non-silent intervals using decibel threshold relative to peak
    raw_intervals = librosa.effects.split(
        waveform, top_db=top_db, frame_length=1024, hop_length=256
    )

    pad_samples = int(pad_sec * sr)
    min_samples = int(min_speech_sec * sr)

    voiced_chunks: List[np.ndarray] = []
    voiced_intervals_sec: List[Tuple[float, float]] = []

    for start_idx, end_idx in raw_intervals:
        if (end_idx - start_idx) < min_samples:
            continue
        p_start = max(0, int(start_idx) - pad_samples)
        p_end = min(raw_len, int(end_idx) + pad_samples)
        voiced_chunks.append(waveform[p_start:p_end])
        voiced_intervals_sec.append(
            (round(p_start / float(sr), 3), round(p_end / float(sr), 3))
        )

    if not voiced_chunks:
        return {
            "cleaned_waveform": np.array([], dtype=np.float32),
            "voiced_intervals_sec": [],
            "raw_duration_sec": round(raw_duration_sec, 3),
            "voiced_duration_sec": 0.0,
            "silence_removed_sec": round(raw_duration_sec, 3),
            "speech_ratio": 0.0,
            "is_silent": True,
        }

    cleaned_waveform = np.concatenate(voiced_chunks).astype(np.float32)
    voiced_duration_sec = len(cleaned_waveform) / float(sr)
    silence_removed_sec = max(0.0, raw_duration_sec - voiced_duration_sec)
    speech_ratio = min(1.0, voiced_duration_sec / max(raw_duration_sec, 1e-6))

    return {
        "cleaned_waveform": cleaned_waveform,
        "voiced_intervals_sec": voiced_intervals_sec,
        "raw_duration_sec": round(raw_duration_sec, 3),
        "voiced_duration_sec": round(voiced_duration_sec, 3),
        "silence_removed_sec": round(silence_removed_sec, 3),
        "speech_ratio": round(speech_ratio, 4),
        "is_silent": False,
    }


def record_from_microphone(
    duration_sec: float = 5.0,
    sr: int = SAMPLE_RATE,
    save_path: Union[str, Path, None] = None,
) -> np.ndarray:
    """
    Record live audio directly from the default system microphone using `sounddevice`.

    Parameters
    ----------
    duration_sec : float, default=5.0
        Recording length in seconds.
    sr : int, default=16000
        Sampling rate in Hz (16 kHz for Whisper).
    save_path : str, Path, or None
        Optional path to save the recorded WAV file.

    Returns
    -------
    waveform : np.ndarray of shape (num_samples,), dtype=float32
    """
    import sounddevice as sd

    num_frames = int(round(duration_sec * sr))
    recording = sd.rec(num_frames, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    waveform = recording.flatten().astype(np.float32)

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), waveform, sr)

    return waveform
