r"""
Phase 4 Verification Script (`scripts/test_vad_mic.py`)
Run with:
    .\.venv\Scripts\python.exe scripts/test_vad_mic.py

Tests Voice Activity Detection (`apply_vad_filter`) in two realistic scenarios:
1. Speech surrounded by 6.0 seconds of dead silence (3s before + 3s after) -> verifies silence trimming.
2. Pure background room noise (5.0 seconds of low-amplitude noise) -> verifies `is_silent=True` gate
   blocks Whisper from hallucinating phantom text.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.audio_utils import SAMPLE_RATE, load_and_preprocess_audio
from src.vad_mic import apply_vad_filter

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def main() -> None:
    print("=" * 70)
    print("  Testing Voice Activity Detection (VAD) — `src/vad_mic.py`")
    print("=" * 70)

    # Test 1: Speech padded with 3.0s leading silence + 3.0s trailing silence
    en_wave = load_and_preprocess_audio(SAMPLES_DIR / "sample_en.wav")
    silence_3s = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)
    padded_wave = np.concatenate([silence_3s, en_wave, silence_3s])

    res1 = apply_vad_filter(padded_wave, top_db=28.0)
    print("\n[Test 1: Speech + 6.0s of Dead Air]")
    print(f"  Raw Recording Duration : {res1['raw_duration_sec']}s")
    print(f"  Voiced Speech Kept     : {res1['voiced_duration_sec']}s")
    print(f"  Silence Removed        : {res1['silence_removed_sec']}s")
    print(f"  Speech Ratio           : {res1['speech_ratio'] * 100:.1f}%")
    print(f"  Voiced Intervals (sec) : {res1['voiced_intervals_sec']}")
    print(f"  Is Silent Gate         : {res1['is_silent']}")
    assert not res1["is_silent"], "Expected voiced speech to pass VAD gate!"
    assert res1["silence_removed_sec"] >= 5.5, "Expected at least 5.5s of silence removed!"

    # Test 2: 5.0 seconds of quiet room noise (no human speech)
    rng = np.random.default_rng(42)
    room_noise = rng.normal(0.0, 0.0005, size=5 * SAMPLE_RATE).astype(np.float32)
    res2 = apply_vad_filter(room_noise, top_db=28.0)
    print("\n[Test 2: 5.0s Quiet Room Noise (No Speech)]")
    print(f"  Raw Recording Duration : {res2['raw_duration_sec']}s")
    print(f"  Voiced Speech Kept     : {res2['voiced_duration_sec']}s")
    print(f"  Is Silent Gate         : {res2['is_silent']} (Blocks Whisper hallucination!)")
    assert res2["is_silent"], "Expected room noise to be flagged as silent!"

    print("\n" + "=" * 70)
    print("SUCCESS: `src/vad_mic.py` passed all VAD silence trimming & gate tests!")


if __name__ == "__main__":
    main()
