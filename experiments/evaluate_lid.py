r"""
Phase 5 (Part 1) — Spoken Language Identification Classification Benchmark (`experiments/evaluate_lid.py`)
Run with:
    .\.venv\Scripts\python.exe experiments/evaluate_lid.py

Satisfies the Yapay Zekâ Dersi classification requirement:
"Sınıflandırma çalışmalarında Accuracy, Precision, Recall, F1-Score ve ROC-AUC metriklerinin sunulması."

Compares 4 models on the 40 unseen test clips (20 English + 20 French):
1. Custom MFCC + Logistic Regression
2. Custom MFCC + Random Forest
3. Custom MFCC + MLP Neural Network
4. OpenAI Whisper (`tiny`) Zero-Shot Encoder Language Identification

Outputs:
- `models/lid_mfcc_classifier.joblib` (best custom classifier for `app.py`)
- `outputs/lid_classification_metrics.csv` (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- `outputs/figures/lid_classification_evaluation.png` (ROC-AUC Curves + Confusion Matrices)
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
from sklearn.metrics import roc_curve

from src.asr_engine import detect_spoken_language, load_whisper_model
from src.audio_utils import SAMPLE_RATE, load_and_preprocess_audio
from src.lid_classifier import (
    augment_acoustic_channel,
    evaluate_binary_predictions,
    train_and_select_lid_models,
)

METRICS_CSV = PROJECT_ROOT / "outputs" / "lid_classification_metrics.csv"
EVAL_FIG = PROJECT_ROOT / "outputs" / "figures" / "lid_classification_evaluation.png"


def main() -> None:
    print("=" * 80)
    print("  VoiceScribe — Language Identification Classification Benchmark (EN vs FR)")
    print("=" * 80)

    # 1. Train & evaluate the 3 custom MFCC Scikit-Learn models
    train_summary = train_and_select_lid_models()
    all_results = train_summary["all_results"]
    y_test = train_summary["y_test"]
    test_df = train_summary["test_df"]

    # 2. Evaluate Whisper (`tiny`) zero-shot Language ID on the exact same 40 test clips
    print("Evaluating local Whisper (`tiny`) zero-shot Language ID on the 40 test clips...")
    whisper_model = load_whisper_model("tiny", device="cpu")
    w_preds = []
    w_probs_fr = []

    for idx, row in test_df.iterrows():
        wav_path = PROJECT_ROOT / str(row["file_path"])
        waveform = load_and_preprocess_audio(wav_path, target_sr=SAMPLE_RATE)
        noisy_wave = augment_acoustic_channel(waveform, seed=5000 + int(row.name), is_test=True)
        lid_info = detect_spoken_language(whisper_model, noisy_wave)
        prob_fr = float(lid_info["en_vs_fr_probs"]["fr"])
        pred_label = 1 if lid_info["predicted_lang"] == "fr" else 0
        w_preds.append(pred_label)
        w_probs_fr.append(prob_fr)

    w_preds_arr = np.array(w_preds, dtype=int)
    w_probs_arr = np.array(w_probs_fr, dtype=float)
    w_metrics = evaluate_binary_predictions(y_test, w_preds_arr, w_probs_arr)
    w_metrics["y_pred"] = w_preds_arr
    w_metrics["y_prob_fr"] = w_probs_arr
    all_results["Whisper Transformer (tiny)"] = w_metrics

    # 3. Build comparison table
    table_rows = []
    for model_name, m in all_results.items():
        cm = m["Confusion_Matrix"]
        table_rows.append(
            {
                "Model": model_name,
                "Accuracy": m["Accuracy"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1-Score": m["F1-Score"],
                "ROC-AUC": m["ROC-AUC"],
                "TN (EN->EN)": cm[0][0],
                "FP (EN->FR)": cm[0][1],
                "FN (FR->EN)": cm[1][0],
                "TP (FR->FR)": cm[1][1],
            }
        )

    metrics_df = pd.DataFrame(table_rows)
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV, index=False)

    print("\n--- Classification Results on Held-Out Test Set (N = 40: 20 EN + 20 FR) ---")
    print(metrics_df.to_string(index=False))
    print(f"\nBest Custom Model Saved : {train_summary['best_model_name']} -> {train_summary['saved_path'].relative_to(PROJECT_ROOT)}")

    # 4. Plot ROC-AUC Curves + Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    fig.suptitle(
        "Spoken Language Identification (English vs. French) — ROC-AUC & Confusion Matrices (Test N=40)",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: ROC Curves for all 4 models
    colors = ["#2563eb", "#059669", "#7c3aed", "#dc2626"]
    for (model_name, m), col in zip(all_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, m["y_prob_fr"])
        axes[0].plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {m['ROC-AUC']:.3f})",
            linewidth=2.0,
            color=col,
        )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Chance (0.500)")
    axes[0].set_title("1. ROC Curves (All 4 Models)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0].legend(loc="lower right", fontsize=8.2)
    axes[0].grid(alpha=0.3)

    # Panel 2: Best Custom MFCC Model Confusion Matrix
    best_custom_name = train_summary["best_model_name"]
    best_cm = np.array(all_results[best_custom_name]["Confusion_Matrix"])
    im1 = axes[1].imshow(best_cm, cmap="Blues")
    axes[1].set_title(f"2. {best_custom_name}\nConfusion Matrix", fontweight="bold")
    axes[1].set_xticks([0, 1], ["Pred: EN (0)", "Pred: FR (1)"])
    axes[1].set_yticks([0, 1], ["True: EN (0)", "True: FR (1)"])
    for i in range(2):
        for j in range(2):
            axes[1].text(
                j,
                i,
                str(best_cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if best_cm[i, j] > 10 else "#0f172a",
            )
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    # Panel 3: Whisper Transformer Confusion Matrix
    w_cm = np.array(w_metrics["Confusion_Matrix"])
    im2 = axes[2].imshow(w_cm, cmap="Greens")
    axes[2].set_title("3. Whisper Transformer (tiny)\nConfusion Matrix", fontweight="bold")
    axes[2].set_xticks([0, 1], ["Pred: EN (0)", "Pred: FR (1)"])
    axes[2].set_yticks([0, 1], ["True: EN (0)", "True: FR (1)"])
    for i in range(2):
        for j in range(2):
            axes[2].text(
                j,
                i,
                str(w_cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if w_cm[i, j] > 10 else "#0f172a",
            )
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    EVAL_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(EVAL_FIG, dpi=160)
    plt.close(fig)

    print(f"Saved Metrics CSV : {METRICS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved ROC/CM Plot : {EVAL_FIG.relative_to(PROJECT_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
