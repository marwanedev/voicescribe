# VoiceScribe — MVP-First Project Timeline & Teammate Roadmap

**Target Languages:** English (`en`) & French (`fr`)  
**Strategy:** Build a rock-solid **MVP** first (all core functionalities working end-to-end on audio files), then add **Live Microphone Transcription**, then run **Experiments** and finalize the **Report & Presentation**.

---

## Phase 0: Environment, GitHub & Teammate Onboarding (Day 1)
- **Tasks:**
  - Connect local repository to `https://github.com/marwanedev/voicescribe.git` and push `main`.
  - Read [docs/VoiceScribe_Teammate_Guide.pdf](docs/VoiceScribe_Teammate_Guide.pdf) together so both teammates understand every module, UI link, and course rule.
  - Set up Python `.venv` and verify `ffmpeg`, `torch`, `openai-whisper`, `librosa`, `scikit-learn`, `jiwer`, `streamlit`, `matplotlib`.
- **Done when:** Both teammates can pull the repo, activate `.venv`, and import all libraries without errors.

---

## Phase 1: Literature Review & Research Framing (Days 2–4)
- **Course Rule:** 20–30 academic papers summarized; clear problem statement & research question.
- **Tasks:**
  - Divide papers across teammates (10–15 papers each):
    - *Speech Features & Classical ASR:* MFCCs, HMM-GMM, CTC (Graves et al., 2006).
    - *Deep & Self-Supervised ASR:* LAS, Transformer ASR, Wav2Vec 2.0 (Baevski et al., 2020), Conformer (Gulati et al., 2020).
    - *Multilingual ASR & Benchmarks:* OpenAI Whisper (Radford et al., 2022), Google FLEURS (Conneau et al., 2022), Mozilla Common Voice (Ardila et al., 2020), Faster-Whisper / quantization.
    - *Language ID, VAD & Noise:* Spoken Language Identification, Silero VAD, noisy ASR robustness.
  - Build a shared summary table (Paper, Year, Dataset, Method, Key Metric, Limitation).
- **Done when:** Both teammates can explain why Whisper (Encoder-Decoder Transformer) and MFCCs work.

---

## Phase 2: English & French Dataset + EDA (Days 5–6)
- **Course Rule:** Dataset source, sample count, class distribution, descriptive statistics (mean, median, mode, std), and visualizations.
- **Tasks:**
  - Curate a balanced **English + French** evaluation dataset (e.g., 100 EN + 100 FR clips from Google FLEURS or Mozilla Common Voice).
  - Build `experiments/eda.py` to compute and plot:
    - Class balance (`English` vs `French` sample counts).
    - Audio duration statistics (**mean, median, mode, min, max, std**) + histogram/boxplot.
    - Transcript word count distribution.
    - Visual comparison of a raw waveform, 80-bin Log-Mel Spectrogram, and MFCC features.
- **Done when:** EDA charts and summary table are saved and understood by both teammates.

---

## Phase 3: MVP — Core End-to-End Working System (Days 7–11)
*Goal: Every core VoiceScribe functionality working cleanly on uploaded audio (`MP3`, `WAV`, `MP4`).*
- **Step 3.1 — Audio Preprocessing (`src/audio_utils.py`):**
  - Load `MP3`/`WAV`/`MP4` via FFmpeg, convert to `16 kHz` mono `float32`, extract Log-Mel Spectrogram & MFCCs.
- **Step 3.2 — Custom EN/FR Language Classifier (`src/lid_classifier.py`):**
  - Train a local Scikit-Learn classifier (Random Forest / Logistic Regression / MLP) on MFCCs to classify English vs. French.
- **Step 3.3 — Whisper ASR Engine (`src/asr_engine.py`):**
  - Run local Whisper (`tiny` / `base` / `small`), extract language probabilities (`en` vs `fr`), and decode speech into timestamped segments (`beam_size`, `temperature`).
- **Step 3.4 — Subtitle Generator (`src/subtitle_utils.py`):**
  - Convert Whisper segment timestamps into valid `.srt` (`HH:MM:SS,mmm --> HH:MM:SS,mmm`) and plain `.txt`.
- **Step 3.5 — MVP Streamlit Web UI (`app.py`):**
  - Sidebar controls (model size, decoding parameters, EN/FR auto-detect or manual override).
  - File upload (`MP3`, `WAV`, `MP4`), progress bar, detected language confidence card, transcript viewer, and `.srt`/`.txt` download buttons.
- **Done when:** You can upload an English or French audio/video file in Streamlit, see accurate EN/FR detection, read the transcript, and download working `.srt` and `.txt` files.

---

## Phase 4: v2 Upgrade — Live Microphone Transcription & VAD (Days 12–14)
*Goal: Add real-time voice recording on top of the working MVP.*
- **Tasks:**
  - Build `src/vad_mic.py`: capture `16 kHz` microphone frames and use Voice Activity Detection (VAD) to detect speech start/pause and avoid silence hallucinations.
  - Add a **"Live Microphone"** tab to `app.py` that streams spoken phrases into `src/asr_engine.py` and updates live captions + `.srt` timestamps in real time.
- **Done when:** Speaking in English or French into the laptop mic produces live timestamped captions cleanly.

---

## Phase 5: Experiments & Quantitative Evaluation (Days 15–17)
- **Course Rule:** Model comparisons, hyperparameter changes, classification metrics (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`), and literature comparison.
- **Experiments:**
  1. **ASR Benchmark (`experiments/evaluate_asr.py`):** Compare `whisper-tiny` vs `whisper-base` vs `whisper-small` on EN & FR clips using **WER, CER, and Real-Time Factor (RTF)**.
  2. **Language ID Classification (`experiments/evaluate_lid.py`):** Compare your **custom MFCC classifier** vs **Whisper zero-shot Language ID** on EN vs FR using **Accuracy, Precision, Recall, F1-Score, ROC-AUC curve, and Confusion Matrix**.
  3. **Hyperparameter & Noise Ablation:** Compare `beam_size=1` (greedy) vs `beam_size=5`, and clean vs noisy audio (with/without VAD).
- **Done when:** All result tables and charts are generated and reproducible.

---

## Phase 6: Academic Report, Presentation & Defense Prep (Days 18–21)
- **Deliverables:** Full Academic Report, Source Code, and Presentation Slides.
- **Tasks:**
  - Write all required academic sections in **your own words** (Abstract, Introduction, Literature Review, Experimental Methods, Results, Discussion, Conclusion, References).
  - Prepare presentation slides covering all required sections + live demo video/backup.
  - Practice Q&A so both teammates can explain MFCCs, Log-Mel Spectrograms, Whisper's Encoder-Decoder, WER vs. ROC-AUC, and limitations.
