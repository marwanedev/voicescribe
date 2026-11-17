"""
Step 3.3 — Local Whisper ASR & Language Identification Engine (`src/asr_engine.py`)

This module runs OpenAI's Whisper Encoder-Decoder Transformer 100% locally on CPU:
1. Loads and caches Whisper model weights (`tiny` = 39M, `base` = 74M, `small` = 244M).
2. Performs Spoken Language Identification (`detect_spoken_language`) by passing the 30-second
   80-bin Log-Mel Spectrogram through the Transformer Encoder and reading language token
   probabilities (`<|en|>` vs `<|fr|>`), providing exact probability scores for ROC-AUC.
3. Transcribes (or translates) audio with configurable hyperparameters (`beam_size`,
   `temperature`) and calculates the Real-Time Factor (RTF = inference_time / audio_duration).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import whisper

from src.audio_utils import SAMPLE_RATE

# In-memory cache so models are loaded once per session
_MODEL_CACHE: Dict[str, Any] = {}

SUPPORTED_MODELS: Tuple[str, ...] = ("tiny", "base", "small", "medium")
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "fr": "French",
}


def load_whisper_model(model_size: str = "tiny", device: str = "cpu") -> Any:
    """
    Load (and cache) a local OpenAI Whisper model.

    Model Parameter Counts (for Presentation & Report):
    - `tiny`  :  39M parameters (4 encoder layers, 4 decoder layers) -> fastest on CPU
    - `base`  :  74M parameters (6 encoder layers, 6 decoder layers) -> balanced accuracy/speed
    - `small` : 244M parameters (12 encoder layers, 12 decoder layers) -> high accuracy
    """
    if model_size not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_size '{model_size}'. Choose from {SUPPORTED_MODELS}."
        )

    cache_key = f"{model_size}:{device}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = whisper.load_model(model_size, device=device)
    return _MODEL_CACHE[cache_key]


def detect_spoken_language(
    model: Any,
    waveform: np.ndarray,
    allowed_langs: Tuple[str, ...] = ("en", "fr"),
) -> Dict[str, Any]:
    """
    Use Whisper's Transformer Encoder to detect the spoken language and return probability
    scores suitable for both UI display and binary/multi-class ROC-AUC evaluation.

    How It Works Under the Hood:
    1. Pads/trims waveform to 30 seconds (480,000 samples at 16 kHz).
    2. Converts waveform to an (80, 3000) Log-Mel Spectrogram tensor.
    3. Passes spectrogram into `model.detect_language(mel)` to get softmax probabilities
       over all 99 language tokens (`<|en|>`, `<|fr|>`, etc.).
    4. Normalizes probabilities over `allowed_langs` (`en` and `fr`) so P(en) + P(fr) = 1.0.
    """
    audio_30s = whisper.pad_or_trim(waveform)
    mel = whisper.log_mel_spectrogram(audio_30s).to(model.device)

    _, all_probs = model.detect_language(mel)

    # Top language across all 99 languages Whisper knows
    raw_top_lang = max(all_probs, key=all_probs.get)
    raw_top_prob = float(all_probs[raw_top_lang])

    # Restrict & normalize over our project languages (English vs French)
    subset_raw = {lang: float(all_probs.get(lang, 0.0)) for lang in allowed_langs}
    total_subset = sum(subset_raw.values())
    if total_subset > 0:
        normalized_probs = {k: v / total_subset for k, v in subset_raw.items()}
    else:
        normalized_probs = {k: 1.0 / len(allowed_langs) for k in allowed_langs}

    predicted_lang = max(normalized_probs, key=normalized_probs.get)
    confidence = float(normalized_probs[predicted_lang])

    return {
        "predicted_lang": predicted_lang,
        "predicted_lang_name": LANGUAGE_NAMES.get(predicted_lang, predicted_lang),
        "confidence": round(confidence, 4),
        "en_vs_fr_probs": {k: round(v, 4) for k, v in normalized_probs.items()},
        "raw_probs_en_fr": {k: round(v, 4) for k, v in subset_raw.items()},
        "raw_top_lang": raw_top_lang,
        "raw_top_prob": round(raw_top_prob, 4),
    }


def transcribe_audio(
    model: Any,
    waveform: np.ndarray,
    language: Optional[str] = None,
    beam_size: int = 1,
    temperature: float = 0.0,
    task: str = "transcribe",
) -> Dict[str, Any]:
    """
    Run Whisper speech-to-text decoding on a 16 kHz mono waveform and compute timing metrics.

    Parameters
    ----------
    model : whisper.Whisper
        Loaded Whisper model instance.
    waveform : np.ndarray
        16 kHz mono float32 waveform.
    language : str or None, default=None
        If None, automatically detects English ('en') or French ('fr') first.
        If 'en' or 'fr', forces decoding in that language.
    beam_size : int, default=1
        Decoding strategy: 1 = Greedy search (fastest); >1 (e.g. 5) = Beam Search.
    temperature : float, default=0.0
        Sampling temperature (0.0 = deterministic argmax/beam decoding).
    task : str, default="transcribe"
        "transcribe" keeps spoken language; "translate" translates French -> English.

    Returns
    -------
    result : dict
        Contains `text`, `segments`, `language_info`, `audio_duration_sec`,
        `inference_time_sec`, and `rtf` (Real-Time Factor).
    """
    audio_duration_sec = max(len(waveform) / float(SAMPLE_RATE), 1e-3)

    t0 = time.perf_counter()

    # Step 1: Run Language Identification
    lang_info = detect_spoken_language(model, waveform)
    chosen_lang = language if language in ("en", "fr") else lang_info["predicted_lang"]

    # Step 2: Configure decoding options (fp16=False because CPU uses float32)
    decode_kwargs: Dict[str, Any] = {
        "language": chosen_lang,
        "task": task,
        "temperature": temperature,
        "fp16": False,
    }
    if beam_size > 1:
        decode_kwargs["beam_size"] = beam_size

    raw_result = model.transcribe(waveform, **decode_kwargs)
    inference_time_sec = time.perf_counter() - t0
    rtf = inference_time_sec / audio_duration_sec

    # Clean segment list for subtitle export
    clean_segments: List[Dict[str, Any]] = []
    for idx, seg in enumerate(raw_result.get("segments", []), start=1):
        clean_segments.append(
            {
                "id": idx,
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "text": str(seg["text"]).strip(),
            }
        )

    return {
        "text": str(raw_result.get("text", "")).strip(),
        "segments": clean_segments,
        "language": chosen_lang,
        "language_name": LANGUAGE_NAMES.get(chosen_lang, chosen_lang),
        "language_info": lang_info,
        "beam_size": beam_size,
        "temperature": temperature,
        "task": task,
        "audio_duration_sec": round(audio_duration_sec, 3),
        "inference_time_sec": round(inference_time_sec, 3),
        "rtf": round(rtf, 3),
    }
