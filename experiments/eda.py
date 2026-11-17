r"""
Phase 2 — Exploratory Data Analysis & Descriptive Statistics (`experiments/eda.py`)
Run with:
    .\.venv\Scripts\python.exe experiments/eda.py

Satisfies the Yapay Zekâ Dersi rubric requirement:
"Veri setinin tanıtılması: Veri setinin kaynağı, özellikleri, örnek sayısı ve sınıf dağılımı;
uygun grafikler ve uygulanabilir tanımlayıcı istatistikler (ortalama, medyan, mod vb.)."

Computes:
- Sample Count & Class Distribution (English vs. French, Train vs. Test)
- Mean (Ortalama), Median (Medyan), Mode (Mod), Standard Deviation, Min, Max for:
  1. Audio Duration (seconds)
  2. Word Count per Sentence
  3. Speaking Rate (Words / Second)
  4. Signal RMS Energy
- Saves `outputs/eda_summary_stats.csv` and `outputs/figures/dataset_eda_overview.png`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
STATS_CSV = PROJECT_ROOT / "outputs" / "eda_summary_stats.csv"
EDA_FIG = PROJECT_ROOT / "outputs" / "figures" / "dataset_eda_overview.png"


def compute_descriptive_row(series: pd.Series, metric_name: str, group_name: str) -> dict:
    """Compute Mean, Median, Mode, Std, Min, and Max for a numerical column."""
    clean = series.dropna().astype(float)
    # For continuous variables, round to 1 decimal place when computing empirical mode
    rounded = clean.round(1) if clean.dtype == float else clean
    mode_res = sp_stats.mode(rounded, keepdims=True)
    mode_val = float(mode_res.mode[0]) if len(mode_res.mode) > 0 else float("nan")

    return {
        "Group": group_name,
        "Metric": metric_name,
        "Count (N)": int(len(clean)),
        "Mean (Ortalama)": round(float(clean.mean()), 3),
        "Median (Medyan)": round(float(clean.median()), 3),
        "Mode (Mod)": round(mode_val, 3),
        "Std Dev (Std)": round(float(clean.std()), 3),
        "Min": round(float(clean.min()), 3),
        "Max": round(float(clean.max()), 3),
    }


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"{METADATA_CSV} not found. Run `python scripts/build_dataset.py` first."
        )

    df = pd.read_csv(METADATA_CSV)

    print("=" * 78)
    print("  VoiceScribe — Dataset Introduction & Descriptive Statistics (Phase 2 EDA)")
    print("=" * 78)
    print(f"Total Samples : {len(df)}")
    print(f"Class Balance : {df['language_name'].value_counts().to_dict()}")
    print(f"Train / Test  : {df['split'].value_counts().to_dict()}")

    metrics_to_summarize = [
        ("duration_sec", "Audio Duration (s)"),
        ("word_count", "Word Count (words)"),
        ("speaking_rate_wps", "Speaking Rate (words/s)"),
        ("rms_energy", "RMS Energy (amplitude)"),
    ]

    rows = []
    for col, label in metrics_to_summarize:
        rows.append(compute_descriptive_row(df[col], label, "All Dataset (N=200)"))
        for lang_code, lang_label in [("en", "English (N=100)"), ("fr", "French (N=100)")]:
            sub = df[df["language"] == lang_code][col]
            rows.append(compute_descriptive_row(sub, label, lang_label))

    stats_df = pd.DataFrame(rows)
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(STATS_CSV, index=False)

    print("\n--- Descriptive Statistics Table (Mean, Median, Mode, Std, Min, Max) ---")
    print(stats_df.to_string(index=False))

    # Build 4-panel publication figure for report & presentation slides
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle(
        "VoiceScribe Bilingual Dataset (N = 200) — Exploratory Data Analysis & Statistics",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Class & Split Distribution
    split_counts = df.groupby(["language_name", "split"]).size().unstack(fill_value=0)
    split_counts[["train", "test"]].plot(
        kind="bar",
        stacked=True,
        ax=axes[0, 0],
        color=["#2563eb", "#93c5fd"],
        edgecolor="#1e293b",
        rot=0,
    )
    axes[0, 0].set_title("1. Class Distribution & Train/Test Split (50% EN / 50% FR)", fontweight="bold")
    axes[0, 0].set_xlabel("Language Class")
    axes[0, 0].set_ylabel("Number of Audio Clips")
    axes[0, 0].legend(["Train Split (80%)", "Test Split (20%)"])
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Panel 2: Audio Duration Distribution by Language
    en_dur = df[df["language"] == "en"]["duration_sec"]
    fr_dur = df[df["language"] == "fr"]["duration_sec"]
    axes[0, 1].hist(
        [en_dur, fr_dur],
        bins=14,
        label=["English (EN)", "French (FR)"],
        color=["#2563eb", "#dc2626"],
        alpha=0.78,
        edgecolor="#1e293b",
    )
    axes[0, 1].axvline(df["duration_sec"].mean(), color="#0f172a", linestyle="--", linewidth=1.5, label=f"Mean = {df['duration_sec'].mean():.2f}s")
    axes[0, 1].axvline(df["duration_sec"].median(), color="#059669", linestyle=":", linewidth=1.5, label=f"Median = {df['duration_sec'].median():.2f}s")
    axes[0, 1].set_title("2. Audio Duration Histogram with Mean & Median", fontweight="bold")
    axes[0, 1].set_xlabel("Duration (seconds)")
    axes[0, 1].set_ylabel("Clip Count")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Panel 3: Boxplot of Speaking Rate (words/sec) by Language
    data_box = [
        df[df["language"] == "en"]["speaking_rate_wps"].values,
        df[df["language"] == "fr"]["speaking_rate_wps"].values,
    ]
    bp = axes[1, 0].boxplot(
        data_box,
        tick_labels=["English (EN)", "French (FR)"],
        patch_artist=True,
        widths=0.45,
    )
    for patch, color in zip(bp["boxes"], ["#bfdbfe", "#fecaca"]):
        patch.set_facecolor(color)
    axes[1, 0].set_title("3. Speaking Rate Boxplot (Words / Second)", fontweight="bold")
    axes[1, 0].set_ylabel("Words per Second (WPS)")
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Panel 4: Scatter of Duration vs Word Count
    for lang_code, color, label in [("en", "#2563eb", "English"), ("fr", "#dc2626", "French")]:
        sub = df[df["language"] == lang_code]
        axes[1, 1].scatter(
            sub["word_count"],
            sub["duration_sec"],
            alpha=0.65,
            color=color,
            label=label,
            edgecolors="w",
            s=48,
        )
    axes[1, 1].set_title("4. Sentence Length (Words) vs. Audio Duration (s)", fontweight="bold")
    axes[1, 1].set_xlabel("Word Count per Sentence")
    axes[1, 1].set_ylabel("Audio Duration (s)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    EDA_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(EDA_FIG, dpi=160)
    plt.close(fig)

    print(f"\nSaved Summary CSV : {STATS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved EDA Figure  : {EDA_FIG.relative_to(PROJECT_ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
