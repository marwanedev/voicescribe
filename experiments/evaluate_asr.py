r"""
Phase 5 (Part 2) — Whisper ASR Model Scale, Hyperparameter & Noise Benchmark (`experiments/evaluate_asr.py`)
Run with:
    .\.venv\Scripts\python.exe experiments/evaluate_asr.py

Evaluates local Whisper models on the held-out Test Set (N = 40: 20 English + 20 French)
across 5 controlled experimental configurations:
1. `tiny`  (39M params)  | `beam_size=1` (Greedy)      | Clean Audio
2. `tiny`  (39M params)  | `beam_size=5` (Beam Search) | Clean Audio  [Hyperparameter Tuning]
3. `base`  (74M params)  | `beam_size=1` (Greedy)      | Clean Audio
4. `small` (244M params) | `beam_size=1` (Greedy)      | Clean Audio
5. `base`  (74M params)  | `beam_size=1` (Greedy)      | Noisy Audio (8 dB SNR) [Noise Ablation]

Outputs:
- `outputs/asr_benchmark_summary.csv`
- `outputs/asr_detailed_predictions.csv`
- `outputs/figures/asr_benchmark_comparison.png`
"""

import gc
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jiwer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.asr_engine import load_whisper_model, transcribe_audio
from src.audio_utils import SAMPLE_RATE, load_and_preprocess_audio

METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
SUMMARY_CSV = PROJECT_ROOT / "outputs" / "asr_benchmark_summary.csv"
DETAILS_CSV = PROJECT_ROOT / "outputs" / "asr_detailed_predictions.csv"
BENCHMARK_FIG = PROJECT_ROOT / "outputs" / "figures" / "asr_benchmark_comparison.png"

_NORM_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)


def normalize_asr_text(text: str) -> str:
    """
    Standardize English and French text before computing WER/CER:
    - Expands apostrophes and hyphens (e.g., French `l'intelligence` -> `l intelligence`,
      `quatre-vingts` -> `quatre vingts`) into spaces so punctuation tokenization
      differences do not artificially inflate Word Error Rate.
    """
    cleaned = (
        str(text)
        .replace("’", " ")
        .replace("'", " ")
        .replace("-", " ")
    )
    return _NORM_TRANSFORM(cleaned)


def add_awgn_noise(waveform: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Add controlled Additive White Gaussian Noise (AWGN) at a target Signal-to-Noise Ratio (dB)."""
    sig_power = float(np.mean(np.square(waveform))) + 1e-8
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return (waveform + noise).astype(np.float32)


def run_configuration(
    cfg: Dict[str, Any],
    test_df: pd.DataFrame,
    waveforms_clean: Dict[str, np.ndarray],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one ASR configuration across all 40 test clips and aggregate EN, FR, and Overall metrics."""
    model_size = cfg["model_size"]
    beam_size = cfg["beam_size"]
    snr_db = cfg["snr_db"]
    label = cfg["label"]

    print(f"\n--> Running [{label}] on N={len(test_df)} test clips...")
    model = load_whisper_model(model_size=model_size, device="cpu")

    per_clip_rows: List[Dict[str, Any]] = []
    for idx, row in test_df.iterrows():
        clip_id = str(row["clip_id"])
        lang = str(row["language"])
        ref_text = str(row["transcript"])
        waveform = waveforms_clean[clip_id]

        if snr_db is not None:
            waveform = add_awgn_noise(waveform, snr_db=snr_db, seed=9000 + int(idx))

        res = transcribe_audio(
            model=model,
            waveform=waveform,
            language=lang,  # Evaluate pure ASR decoding accuracy per language
            beam_size=beam_size,
            temperature=0.0,
            task="transcribe",
        )

        ref_norm = normalize_asr_text(ref_text)
        hyp_norm = normalize_asr_text(res["text"])

        clip_wer = jiwer.wer(ref_norm, hyp_norm) * 100.0 if ref_norm else 0.0
        clip_cer = jiwer.cer(ref_norm, hyp_norm) * 100.0 if ref_norm else 0.0

        per_clip_rows.append(
            {
                "config": label,
                "model_size": model_size,
                "beam_size": beam_size,
                "condition": "Clean" if snr_db is None else f"Noisy ({snr_db} dB)",
                "clip_id": clip_id,
                "language": lang,
                "duration_sec": res["audio_duration_sec"],
                "inference_sec": res["inference_time_sec"],
                "rtf": res["rtf"],
                "wer_pct": round(clip_wer, 2),
                "cer_pct": round(clip_cer, 2),
                "ref_norm": ref_norm,
                "hyp_norm": hyp_norm,
                "hypothesis": res["text"],
            }
        )

    df_cfg = pd.DataFrame(per_clip_rows)

    # Corpus-level WER & CER (standard weighted aggregation across all sentences)
    overall_wer = jiwer.wer(df_cfg["ref_norm"].tolist(), df_cfg["hyp_norm"].tolist()) * 100.0
    overall_cer = jiwer.cer(df_cfg["ref_norm"].tolist(), df_cfg["hyp_norm"].tolist()) * 100.0

    df_en = df_cfg[df_cfg["language"] == "en"]
    df_fr = df_cfg[df_cfg["language"] == "fr"]
    en_wer = jiwer.wer(df_en["ref_norm"].tolist(), df_en["hyp_norm"].tolist()) * 100.0
    fr_wer = jiwer.wer(df_fr["ref_norm"].tolist(), df_fr["hyp_norm"].tolist()) * 100.0
    en_cer = jiwer.cer(df_en["ref_norm"].tolist(), df_en["hyp_norm"].tolist()) * 100.0
    fr_cer = jiwer.cer(df_fr["ref_norm"].tolist(), df_fr["hyp_norm"].tolist()) * 100.0

    summary_row = {
        "Configuration": label,
        "Model": model_size,
        "Params_M": cfg["params_m"],
        "Beam_Size": beam_size,
        "Condition": "Clean" if snr_db is None else f"Noisy ({snr_db} dB)",
        "Overall_WER_%": round(overall_wer, 2),
        "Overall_CER_%": round(overall_cer, 2),
        "EN_WER_%": round(en_wer, 2),
        "FR_WER_%": round(fr_wer, 2),
        "EN_CER_%": round(en_cer, 2),
        "FR_CER_%": round(fr_cer, 2),
        "Avg_Time_s": round(float(df_cfg["inference_sec"].mean()), 3),
        "Mean_RTF": round(float(df_cfg["rtf"].mean()), 3),
    }

    print(
        f"    Done -> Overall WER={summary_row['Overall_WER_%']}% "
        f"(EN={summary_row['EN_WER_%']}%, FR={summary_row['FR_WER_%']}%) | "
        f"CER={summary_row['Overall_CER_%']}% | Mean RTF={summary_row['Mean_RTF']}x"
    )
    return summary_row, per_clip_rows


def plot_benchmark_results(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Create a clean 3-panel chart for the report and presentation slides."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=True)
    fig.suptitle(
        "VoiceScribe ASR Benchmark (Held-Out Test Set N = 40: 20 EN + 20 FR)",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Model Scale WER Comparison (tiny vs base vs small, Clean, Beam=1)
    scale_df = summary_df[
        (summary_df["Beam_Size"] == 1) & (summary_df["Condition"] == "Clean")
    ].copy()
    x = np.arange(len(scale_df))
    w = 0.26
    axes[0].bar(x - w, scale_df["EN_WER_%"], width=w, label="English WER %", color="#2563eb")
    axes[0].bar(x, scale_df["FR_WER_%"], width=w, label="French WER %", color="#dc2626")
    axes[0].bar(x + w, scale_df["Overall_WER_%"], width=w, label="Overall WER %", color="#0f172a")
    axes[0].set_xticks(x, [f"{r['Model']}\n({r['Params_M']}M)" for _, r in scale_df.iterrows()])
    axes[0].set_title("1. Model Scale: WER (Lower = Better)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Word Error Rate (WER %)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Accuracy vs Speed Trade-off (Overall WER vs Mean RTF)
    colors = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626"]
    offsets = [(8, -4), (8, 5), (8, -4), (-78, 10), (8, -4)]
    for (_, row), col, (dx, dy) in zip(summary_df.iterrows(), colors, offsets):
        axes[1].scatter(
            row["Mean_RTF"],
            row["Overall_WER_%"],
            s=130,
            color=col,
            edgecolors="#0f172a",
            zorder=3,
        )
        axes[1].annotate(
            row["Configuration"],
            (row["Mean_RTF"], row["Overall_WER_%"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
        )
    axes[1].axvline(1.0, color="#64748b", linestyle="--", linewidth=1.2, label="Real-Time Threshold (RTF = 1.0x)")
    axes[1].set_xlim(0.2, float(summary_df["Mean_RTF"].max()) + 0.6)
    axes[1].set_ylim(0, float(summary_df["Overall_WER_%"].max()) * 1.12)
    axes[1].set_title("2. Accuracy vs. CPU Speed Trade-Off", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Mean Real-Time Factor (RTF = Inference / Audio)")
    axes[1].set_ylabel("Overall WER (%)")
    axes[1].legend(loc="upper right", fontsize=8.2)
    axes[1].grid(alpha=0.3)

    # Panel 3: Hyperparameter (Beam Search) & Noise Ablation
    ablation_labels = summary_df["Configuration"].tolist()
    y_pos = np.arange(len(ablation_labels))
    bar_colors = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"]
    bars = axes[2].barh(y_pos, summary_df["Overall_CER_%"], color=bar_colors[: len(ablation_labels)])
    axes[2].set_yticks(y_pos, ablation_labels, fontsize=8.5)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, float(summary_df["Overall_CER_%"].max()) * 1.22)
    axes[2].set_title("3. Overall CER (%) Across Configs", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Overall Character Error Rate (CER %)")
    axes[2].grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, summary_df["Overall_CER_%"]):
        axes[2].text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=8.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"{METADATA_CSV} not found. Run `python scripts/build_dataset.py` first."
        )

    df = pd.read_csv(METADATA_CSV)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print("=" * 80)
    print(f"  VoiceScribe — Whisper ASR Experimental Benchmark (Test N = {len(test_df)})")
    print("=" * 80)

    # Pre-load all 40 clean test waveforms once
    waveforms_clean: Dict[str, np.ndarray] = {}
    for _, row in test_df.iterrows():
        clip_id = str(row["clip_id"])
        wav_path = PROJECT_ROOT / str(row["file_path"])
        waveforms_clean[clip_id] = load_and_preprocess_audio(wav_path, target_sr=SAMPLE_RATE)

    configs = [
        {"label": "tiny (beam=1, clean)", "model_size": "tiny", "params_m": 39, "beam_size": 1, "snr_db": None},
        {"label": "tiny (beam=5, clean)", "model_size": "tiny", "params_m": 39, "beam_size": 5, "snr_db": None},
        {"label": "base (beam=1, clean)", "model_size": "base", "params_m": 74, "beam_size": 1, "snr_db": None},
        {"label": "small (beam=1, clean)", "model_size": "small", "params_m": 244, "beam_size": 1, "snr_db": None},
        {"label": "base (beam=1, 8dB noise)", "model_size": "base", "params_m": 74, "beam_size": 1, "snr_db": 8.0},
    ]

    summary_rows: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []

    for cfg in configs:
        s_row, details = run_configuration(cfg, test_df, waveforms_clean)
        summary_rows.append(s_row)
        all_details.extend(details)
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.DataFrame(all_details)

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    details_df.to_csv(DETAILS_CSV, index=False)

    plot_benchmark_results(summary_df, BENCHMARK_FIG)

    print("\n" + "=" * 80)
    print("  ASR Benchmark Summary Table (N = 40 Test Clips)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved Summary CSV   : {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved Detailed CSV  : {DETAILS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved Benchmark Fig : {BENCHMARK_FIG.relative_to(PROJECT_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
