# VoiceScribe — Detailed Project Timeline & Learning Roadmap

This timeline is designed to satisfy **100% of the Yapay Zekâ Dersi project requirements** while ensuring **you understand every module, algorithm, and metric** for your live presentation and Q&A.

---

## Phase 0: Environment & GitHub Setup (Day 1)
- **Tasks:**
  - Create Python virtual environment (`.venv`) and install core dependencies (`openai-whisper`, `torch`, `streamlit`, `jiwer`, `scikit-learn`, `librosa`, `matplotlib`).
  - Install `ffmpeg` (required for MP3/MP4 audio decoding).
  - Connect this local Git repository to a remote GitHub repository and push initial files.
- **Checkpoint:** `python -c "import whisper, torch, streamlit"` runs cleanly; repo visible on GitHub.

---

## Phase 1: Research Question & Literature Review (Days 2–4)
- **Course Requirement:** 20–30 academic papers summarized; clear problem statement & research question.
- **Tasks:**
  - Read and organize 20–30 papers into 4 thematic groups:
    1. *Classical & Early Neural ASR:* HMM-GMM, CTC (Graves et al., 2006), DeepSpeech.
    2. *Attention & Self-Supervised ASR:* LAS, Transformer ASR, Wav2Vec 2.0 (Baevski et al., 2020), Conformer (Gulati et al., 2020).
    3. *Multitask & Multilingual ASR:* Whisper (Radford et al., 2022), Google FLEURS (Conneau et al., 2022), Mozilla Common Voice (Ardila et al., 2020), Faster-Whisper / quantization.
    4. *Preprocessing, VAD & Captioning:* Silero VAD, SpecAugment, noisy ASR robustness, SRT segmentation.
  - Build a literature comparison table (Author/Year, Dataset, Method, Key Metric/Result, Limitation).
- **Checkpoint:** You can explain *why* Whisper's multitask Encoder-Decoder Transformer differs from CTC-only models like Wav2Vec 2.0.

---

## Phase 2: Dataset Introduction & Exploratory Data Analysis (Days 5–6)
- **Course Requirement:** Dataset source, features, sample count, class distribution, descriptive stats (mean, median, mode, std), and visualizations.
- **Tasks:**
  - Curate a balanced multilingual evaluation subset (e.g., **Google FLEURS** or **Mozilla Common Voice** across 3–4 languages such as English, Turkish, French, Arabic — e.g., 50–100 clips per language = 200–400 samples).
  - Write `experiments/eda.py` together to compute:
    - Sample count & percentage per language (class distribution chart).
    - Audio duration statistics: **mean, median, mode, min, max, standard deviation** + histogram/boxplot.
    - Transcript word count & speaking rate (words/sec) distributions.
    - Sample Log-Mel Spectrogram visualization for the report/slides.
- **Checkpoint:** EDA plots saved; you can present exact dataset numbers and explain Log-Mel Spectrograms.

---

## Phase 3: Core VoiceScribe Pipeline — Co-Building Module by Module (Days 7–10)
*Rule: We build one small module at a time, inspect every function, and test it before moving on.*
- **Step 3.1 — Audio Preprocessing (`src/audio_utils.py`):**
  - Load MP3/WAV/MP4 via FFmpeg, convert to 16 kHz mono float32 waveform, inspect padding/trimming to 30-second chunks, extract 80-bin Log-Mel Spectrogram.
- **Step 3.2 — Whisper ASR & Language ID (`src/asr_engine.py`):**
  - Load Whisper models (`tiny`, `base`, `small`), inspect encoder features, extract top-k language probabilities (`detect_language`), run `transcribe()` with configurable `beam_size` and `temperature`.
- **Step 3.3 — Timestamped Subtitles (`src/subtitle_utils.py`):**
  - Format Whisper segment timestamps (`start`, `end`) into `HH:MM:SS,mmm` `.srt` blocks and clean `.txt` transcripts.
- **Step 3.4 — Live Microphone & VAD (`src/vad_mic.py`):**
  - Capture microphone audio chunks and filter silence using Voice Activity Detection (to prevent Whisper hallucinations on silent audio).
- **Checkpoint:** End-to-end CLI test on sample audio producing `.srt` and `.txt`, with you able to trace data from waveform -> Mel spectrogram -> tokens -> SRT.

---

## Phase 4: Experimental Studies & Evaluation (Days 11–14)
- **Course Requirement:** Experimental comparison tables/charts, hyperparameter changes, classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC), and literature comparison.
- **Experiments to Run:**
  1. **Model Scale Comparison (`experiments/evaluate_asr.py`):**
     - Test `tiny` (39M), `base` (74M), `small` (244M) on your dataset.
     - Measure **WER** (Word Error Rate), **CER** (Character Error Rate), **RTF** (Real-Time Factor = processing time / audio duration), and **Memory Usage**.
  2. **Language Identification Classification (`experiments/evaluate_lid.py`):**
     - Evaluate Whisper's language classification on your multilingual dataset.
     - Generate **Confusion Matrix** + **Accuracy, Precision, Recall, F1-Score (macro/weighted), and One-vs-Rest ROC-AUC curves**.
  3. **Hyperparameter & Noise Ablation:**
     - Compare Greedy (`beam_size=1`) vs Beam Search (`beam_size=5`).
     - Test robustness under synthetic background noise (e.g., SNR 10 dB / 20 dB) and with/without VAD.
- **Checkpoint:** Clean summary tables and matplotlib/seaborn charts ready for report & slides.

---

## Phase 5: Streamlit Web Application (Days 15–17)
- **Tasks:**
  - Build `app.py` with clean modular imports from `src/`:
    - Sidebar: Model size selector (`tiny`, `base`, `small`), task mode (Transcribe vs Translate), `beam_size` slider, language override/auto-detect.
    - Input tabs: File Upload (MP3, WAV, MP4) & Live Microphone recording.
    - Output panel: Detected language + confidence bar, progress indicator, synchronized transcript viewer, `.srt` and `.txt` download buttons.
- **Checkpoint:** Live demo runs smoothly from start to finish; commit & push final code to GitHub.

---

## Phase 6: Academic Report, Presentation & Defense Prep (Days 18–21)
- **Course Requirement:** Turnitin similarity & **AI detection rate <= 40%**; full academic sections; oral defense readiness.
- **Tasks:**
  - Write the report **in your own words** (use our bulleted outlines and your own experimental tables/figures — never paste raw AI prose into Turnitin).
  - Include all required sections: Abstract (Özet), Introduction (Giriş), Literature Review (Literatür Taraması), Experimental Methods (Deneysel Yöntemler), Results (Bulgular), Discussion (Tartışma), Conclusion (Sonuç), References (Kaynakça).
  - Build presentation slides covering all 8 required presentation sections.
  - Do a mock Q&A session explaining the Transformer encoder-decoder, Mel spectrograms, WER vs ROC-AUC, and limitations.
  - Verify all files (Report + Source Code / GitHub link + Slides) and submit to `yapayzekaproje80@gmail.com`.
