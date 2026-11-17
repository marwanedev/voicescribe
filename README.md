# VoiceScribe: Multilingual Speech-to-Text & Timestamped Subtitle System

**Course:** Yapay Zekâ Dersi (Artificial Intelligence)  
**Deliverables:** Source Code, Academic Report, Presentation (`yapayzekaproje80@gmail.com`)

---

## Project Overview

VoiceScribe is an Automatic Speech Recognition (ASR) and subtitle generation system built around OpenAI's **Whisper** Encoder-Decoder Transformer architecture, paired with an experimental benchmark suite and a **Streamlit** web interface.

### Research Question
> *How do different Whisper model capacities (`tiny`, `base`, `small`) and decoding/preprocessing choices (beam search, VAD filtering, acoustic noise) trade off transcription accuracy (WER/CER), spoken language identification performance (Accuracy, Precision, Recall, F1, ROC-AUC), and real-time inference speed (RTF) across multiple languages?*

---

## Planned Architecture

```text
VOICESCRIBE/
├── src/
│   ├── audio_utils.py       # Audio loading, resampling (16 kHz mono), Log-Mel feature inspection
│   ├── asr_engine.py        # Whisper model loading, language ID, decoding & hyperparameters
│   ├── subtitle_utils.py    # Timestamp formatting and .srt / .txt export
│   └── vad_mic.py           # Voice Activity Detection & live microphone capture
├── experiments/
│   ├── eda.py               # Dataset descriptive statistics (mean, median, mode) & plots
│   ├── evaluate_asr.py      # WER, CER, and Real-Time Factor (RTF) across model sizes
│   └── evaluate_lid.py      # Language ID metrics: Accuracy, Precision, Recall, F1, ROC-AUC
├── app.py                   # Streamlit web application
├── requirements.txt
├── TIMELINE.md
└── README.md
```

---

## Status

- [x] Phase 0: Repository initialized
- [ ] Phase 1: Literature Review (20–30 papers) & Research Question
- [ ] Phase 2: Dataset Selection & Exploratory Data Analysis (EDA)
- [ ] Phase 3: Core ASR Pipeline (built & understood module-by-module)
- [ ] Phase 4: Experiments & Metric Analysis
- [ ] Phase 5: Streamlit UI Integration
- [ ] Phase 6: Academic Report, Presentation & Defense Prep
