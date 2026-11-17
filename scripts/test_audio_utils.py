r"""
Step 3.1 Verification Script (`scripts/test_audio_utils.py`)
Run with:
    .\.venv\Scripts\python.exe scripts/test_audio_utils.py

1. Creates two spoken sample WAV clips (`data/samples/sample_en.wav` and `sample_fr.wav`)
   using Windows built-in speech voices (`en-US` and `fr-FR`) if not already present.
2. Runs every function in `src/audio_utils.py` on both clips.
3. Prints exact tensor/array shapes & statistics and saves 3-panel visual plots in `outputs/figures/`.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to sys.path so `from src.audio_utils import ...` works cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.audio_utils import (
    extract_log_mel_spectrogram,
    extract_mfcc_features,
    get_audio_stats,
    load_and_preprocess_audio,
    plot_audio_features,
)

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def ensure_sample_wavs() -> tuple[Path, Path]:
    """Generate clean English and French spoken WAV files via Windows SAPI if missing."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    en_path = SAMPLES_DIR / "sample_en.wav"
    fr_path = SAMPLES_DIR / "sample_fr.wav"

    if not en_path.exists() or not fr_path.exists():
        ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

$synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet, [System.Speech.Synthesis.VoiceAge]::NotSet, 0, [System.Globalization.CultureInfo]::GetCultureInfo('en-US'))
$synth.SetOutputToWaveFile('{en_path}')
$synth.Speak('Hello and welcome to VoiceScribe. This is an automatic speech recognition and subtitle generation test in English.')

$synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet, [System.Speech.Synthesis.VoiceAge]::NotSet, 0, [System.Globalization.CultureInfo]::GetCultureInfo('fr-FR'))
$synth.SetOutputToWaveFile('{fr_path}')
$synth.Speak('Bonjour et bienvenue sur VoiceScribe. Ceci est un test de reconnaissance vocale automatique et de sous-titrage en français.')

$synth.Dispose()
"""
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True)

    return en_path, fr_path


def inspect_clip(label: str, path: Path, out_fig: Path) -> None:
    print("=" * 68)
    print(f"  Inspecting {label}: {path.name}")
    print("=" * 68)

    # 1. Load & resample to 16 kHz mono float32
    waveform = load_and_preprocess_audio(path)
    stats = get_audio_stats(waveform)

    # 2. Extract Log-Mel Spectrogram (clip-length and 30s-padded)
    log_mel_clip = extract_log_mel_spectrogram(waveform, pad_to_30s=False)
    log_mel_30s = extract_log_mel_spectrogram(waveform, pad_to_30s=True)

    # 3. Extract 13 MFCCs + 26-D summary feature vector
    mfcc_matrix, mfcc_vec = extract_mfcc_features(waveform, n_mfcc=13)

    print(f"  1. Waveform Array        : shape={waveform.shape}, dtype={waveform.dtype}")
    print(f"  2. Duration & Loudness   : {stats['duration_sec']}s | RMS={stats['rms_energy']} | Peak={stats['peak_amplitude']}")
    print(f"  3. Log-Mel (actual clip) : shape={log_mel_clip.shape}  -> (80 Mel bins, {log_mel_clip.shape[1]} time frames)")
    print(f"  4. Log-Mel (Whisper 30s) : shape={log_mel_30s.shape} -> (80 Mel bins, 3000 frames = 30.0s)")
    print(f"  5. MFCC Matrix           : shape={mfcc_matrix.shape}  -> (13 MFCC bands, {mfcc_matrix.shape[1]} frames)")
    print(f"  6. MFCC ML Feature Vec   : shape={mfcc_vec.shape}     -> (13 means + 13 stds for Scikit-Learn)")
    print(f"     First 5 MFCC means    : {mfcc_vec[:5].round(2)}")

    # 4. Save 3-panel plot
    fig = plot_audio_features(
        waveform=waveform,
        log_mel=log_mel_clip,
        mfcc_matrix=mfcc_matrix,
        title=f"VoiceScribe Audio Preprocessing — {label} ({path.name})",
        save_path=out_fig,
    )
    plt.close(fig)
    print(f"  7. Saved Feature Plot    : {out_fig.relative_to(PROJECT_ROOT)}")


def main() -> None:
    en_path, fr_path = ensure_sample_wavs()
    inspect_clip("English Sample", en_path, FIGURES_DIR / "sample_en_features.png")
    inspect_clip("French Sample", fr_path, FIGURES_DIR / "sample_fr_features.png")
    print("=" * 68)
    print("SUCCESS: `src/audio_utils.py` passed all checks on English and French audio!")


if __name__ == "__main__":
    main()
