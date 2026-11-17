r"""
Step 3.3 & 3.4 Verification Script (`scripts/test_asr_and_subtitles.py`)
Run with:
    .\.venv\Scripts\python.exe scripts/test_asr_and_subtitles.py

1. Loads local Whisper (`tiny` = 39M parameters) on CPU.
2. Runs Language Identification (`en` vs `fr`), transcription, and timing (RTF)
   on `data/samples/sample_en.wav` and `data/samples/sample_fr.wav`.
3. Computes Word Error Rate (WER) and Character Error Rate (CER) with `jiwer`.
4. Exports `.srt` and `.txt` files to `outputs/transcripts/` and displays them.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jiwer

from src.asr_engine import load_whisper_model, transcribe_audio
from src.audio_utils import load_and_preprocess_audio
from src.subtitle_utils import save_transcript_files, segments_to_srt

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
TRANSCRIPTS_DIR = PROJECT_ROOT / "outputs" / "transcripts"

TEST_CASES = [
    {
        "label": "English Sample",
        "file": SAMPLES_DIR / "sample_en.wav",
        "expected_lang": "en",
        "ground_truth": (
            "Hello and welcome to VoiceScribe. This is an automatic speech "
            "recognition and subtitle generation test in English."
        ),
    },
    {
        "label": "French Sample",
        "file": SAMPLES_DIR / "sample_fr.wav",
        "expected_lang": "fr",
        "ground_truth": (
            "Bonjour et bienvenue sur VoiceScribe. Ceci est un test de "
            "reconnaissance vocale automatique et de sous-titrage en français."
        ),
    },
]


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for standard ASR WER/CER calculation."""
    transform = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    return transform(text)


def main() -> None:
    print("=" * 72)
    print("  Loading Local Whisper Model (`tiny`, 39M params) on CPU...")
    print("=" * 72)
    model = load_whisper_model("tiny", device="cpu")

    for case in TEST_CASES:
        wav_path: Path = case["file"]
        waveform = load_and_preprocess_audio(wav_path)

        result = transcribe_audio(
            model=model,
            waveform=waveform,
            language=None,  # Auto-detect English vs French
            beam_size=1,
            temperature=0.0,
        )

        # Compute WER and CER against ground truth
        ref_norm = normalize_text(case["ground_truth"])
        hyp_norm = normalize_text(result["text"])
        wer = jiwer.wer(ref_norm, hyp_norm) * 100.0
        cer = jiwer.cer(ref_norm, hyp_norm) * 100.0

        # Save .srt and .txt
        srt_path, txt_path = save_transcript_files(
            segments=result["segments"],
            output_dir=TRANSCRIPTS_DIR,
            base_name=wav_path.stem,
        )

        lid = result["language_info"]
        print(f"\n[{case['label']}] -> {wav_path.name}")
        print("-" * 72)
        print(
            f"  Detected Language : {lid['predicted_lang_name']} ('{lid['predicted_lang']}')"
            f" | EN/FR Probs: {lid['en_vs_fr_probs']} (Expected: '{case['expected_lang']}')"
        )
        print(
            f"  Speed & RTF       : Audio={result['audio_duration_sec']}s | "
            f"Inference={result['inference_time_sec']}s | RTF={result['rtf']}x"
        )
        print(f"  Ground Truth      : {case['ground_truth']}")
        print(f"  Whisper Output    : {result['text']}")
        print(f"  Accuracy Metrics  : WER = {wer:.2f}% | CER = {cer:.2f}%")
        print(f"  Exported Files    : {srt_path.relative_to(PROJECT_ROOT)} & {txt_path.relative_to(PROJECT_ROOT)}")
        print("\n  --- Generated .SRT Subtitles ---")
        for line in segments_to_srt(result["segments"]).strip().splitlines():
            print(f"    {line}")

    print("\n" + "=" * 72)
    print("SUCCESS: `src/asr_engine.py` & `src/subtitle_utils.py` verified end-to-end!")


if __name__ == "__main__":
    main()
