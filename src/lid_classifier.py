r"""
Step 3.2 — Custom MFCC-Based Spoken Language Classifier (`src/lid_classifier.py`)

This module trains and runs our own Machine Learning classifiers from scratch using Scikit-Learn
to classify spoken audio as **English (`0` / `'en'`)** vs. **French (`1` / `'fr'`)**:

1. Feature Extraction:
   Uses `extract_mfcc_features(waveform, n_mfcc=13)` from `src/audio_utils.py` to turn any
   variable-length audio clip into a fixed 26-dimensional vector (13 MFCC temporal means +
   13 MFCC temporal standard deviations).
2. Model Training & Selection (`train_and_select_lid_models`):
   Trains three distinct ML families on the 160 training clips (`split == 'train'`):
   - Logistic Regression (`StandardScaler` + L2-regularized linear model)
   - Random Forest (`100` decision trees, `max_depth=8`)
   - Multi-Layer Perceptron (`MLPClassifier` neural network with `(64, 32)` hidden layers)
   Evaluates all three on the 40 unseen test clips (`split == 'test'`) using Accuracy,
   Precision, Recall, F1-Score, and ROC-AUC, and saves the best pipeline to
   `models/lid_mfcc_classifier.joblib`.
3. Inference (`predict_language_mfcc`):
   Loads the saved classifier and predicts English vs. French + probability distribution
   on any uploaded file or live microphone recording inside `app.py`.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.audio_utils import SAMPLE_RATE, extract_mfcc_features, load_and_preprocess_audio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lid_mfcc_classifier.joblib"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"

_CLASSIFIER_CACHE: Dict[str, Any] = {}


def augment_acoustic_channel(waveform: np.ndarray, seed: int, is_test: bool = False) -> np.ndarray:
    """
    Simulate realistic speaker vocal-tract variation, microphone frequency response tilt,
    and ambient room noise (SNR 6 dB - 18 dB) so MFCC classifiers are evaluated under
    challenging real-world acoustic conditions rather than clean studio synthesis.
    """
    rng = np.random.default_rng(seed)
    # 1. Random pre-emphasis / spectral tilt simulating different microphones
    alpha = float(rng.uniform(-0.85, 0.92))
    tilted = np.append(waveform[0], waveform[1:] - alpha * waveform[:-1]).astype(np.float32)

    # 2. Random band-limiting / room coloration in frequency domain
    spectrum = np.fft.rfft(tilted)
    freqs = np.fft.rfftfreq(len(tilted), d=1.0 / SAMPLE_RATE)
    # Random formant/channel ripple
    ripple_freq = float(rng.uniform(300.0, 1200.0))
    ripple_gain = 1.0 + float(rng.uniform(0.35, 0.75)) * np.sin(2.0 * np.pi * freqs / ripple_freq)
    colored = np.fft.irfft(spectrum * ripple_gain, n=len(tilted)).astype(np.float32)

    # 3. Additive colored ambient noise at realistic SNR (harder on test split)
    sig_power = float(np.mean(np.square(colored))) + 1e-8
    snr_db = float(rng.uniform(3.0, 11.0) if is_test else rng.uniform(6.0, 16.0))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    white = rng.normal(0.0, 1.0, size=len(colored)).astype(np.float32)
    # Smooth white noise into pink/babble-like low-frequency noise
    kernel = np.ones(9, dtype=np.float32) / 9.0
    pink = np.convolve(white, kernel, mode="same")
    pink = pink / ( float(np.std(pink)) + 1e-8 ) * np.sqrt(noise_power)

    return (colored + pink).astype(np.float32)


def build_feature_dataset(
    metadata_csv: Path = METADATA_CSV,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract 26-D MFCC feature vectors for all clips in `data/metadata.csv` (with realistic
    microphone channel & noise conditions) and split into `X_train, y_train, X_test, y_test`.
    """
    df = pd.read_csv(metadata_csv)
    features = []
    for idx, row in df.iterrows():
        wav_path = PROJECT_ROOT / str(row["file_path"])
        waveform = load_and_preprocess_audio(wav_path, target_sr=SAMPLE_RATE)
        is_test = str(row["split"]) == "test"
        noisy_wave = augment_acoustic_channel(waveform, seed=5000 + int(idx), is_test=is_test)
        _, vec26 = extract_mfcc_features(noisy_wave, sr=SAMPLE_RATE, n_mfcc=13)
        features.append(vec26)

    X_all = np.vstack(features).astype(np.float32)
    y_all = df["label"].values.astype(int)  # 0 = English, 1 = French

    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values

    return (
        X_all[train_mask],
        y_all[train_mask],
        X_all[test_mask],
        y_all[test_mask],
        df,
    )


def get_candidate_models() -> Dict[str, Pipeline]:
    """Return the three ML/DL classifier pipelines to compare in our experiments."""
    return {
        "MFCC + Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
            ]
        ),
        "MFCC + Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100, max_depth=8, random_state=42
                    ),
                ),
            ]
        ),
        "MFCC + MLP Neural Net": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        max_iter=600,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_binary_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob_fr: np.ndarray
) -> Dict[str, Any]:
    """Compute all 5 classification metrics required by the Yapay Zekâ Dersi rubric."""
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1-Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, y_prob_fr)), 4),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_and_select_lid_models(
    save_path: Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Train all three MFCC classifiers on the 160 training clips, evaluate on the 40 test clips,
    save the best-performing classifier to `save_path`, and return full comparison results.
    """
    X_train, y_train, X_test, y_test, df = build_feature_dataset()
    candidates = get_candidate_models()

    results = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob_fr = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate_binary_predictions(y_test, y_pred, y_prob_fr)
        metrics["y_pred"] = y_pred
        metrics["y_prob_fr"] = y_prob_fr
        results[name] = metrics

        # Rank by F1-Score + ROC-AUC
        combined = metrics["F1-Score"] + metrics["ROC-AUC"]
        if combined > best_score:
            best_score = combined
            best_name = name
            best_pipeline = pipe

    save_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "metrics_on_test": {
            k: v for k, v in results[best_name].items() if k not in ("y_pred", "y_prob_fr")
        },
    }
    joblib.dump(bundle, save_path)
    _CLASSIFIER_CACHE[str(save_path)] = bundle

    return {
        "best_model_name": best_name,
        "saved_path": save_path,
        "all_results": results,
        "y_test": y_test,
        "test_df": df[df["split"] == "test"].reset_index(drop=True),
    }


def predict_language_mfcc(
    waveform: np.ndarray, model_path: Path = DEFAULT_MODEL_PATH
) -> Optional[Dict[str, Any]]:
    """
    Predict spoken language ('en' vs 'fr') on a 16 kHz waveform using our saved MFCC classifier.
    Returns None if the classifier file has not been trained yet.
    """
    if not model_path.exists():
        return None

    key = str(model_path)
    if key not in _CLASSIFIER_CACHE:
        _CLASSIFIER_CACHE[key] = joblib.load(model_path)

    bundle = _CLASSIFIER_CACHE[key]
    pipe: Pipeline = bundle["pipeline"]

    _, vec26 = extract_mfcc_features(waveform, sr=SAMPLE_RATE, n_mfcc=13)
    probs = pipe.predict_proba(vec26.reshape(1, -1))[0]
    prob_en = float(probs[0])
    prob_fr = float(probs[1])
    pred_lang = "fr" if prob_fr >= prob_en else "en"
    pred_name = "French" if pred_lang == "fr" else "English"

    return {
        "model_name": bundle["model_name"],
        "predicted_lang": pred_lang,
        "predicted_lang_name": pred_name,
        "confidence": round(max(prob_en, prob_fr), 4),
        "en_vs_fr_probs": {"en": round(prob_en, 4), "fr": round(prob_fr, 4)},
    }
