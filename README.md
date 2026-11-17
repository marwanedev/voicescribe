# VoiceScribe: Bilingual (EN/FR) Speech-to-Text & Subtitle System

**Course:** Yapay Zekâ Dersi (Artificial Intelligence)  
**Languages:** English (`en`) & French (`fr`)  
**Repository:** https://github.com/marwanedev/voicescribe  
**Deliverables:** Source Code, Academic Report, Presentation (`yapayzekaproje80@gmail.com`)

---

## Project Overview

VoiceScribe is an **offline** Automatic Speech Recognition (ASR) and timestamped subtitle generation system built with OpenAI's open-source **Whisper** Encoder-Decoder Transformer, a custom **MFCC-based Language Identification classifier**, an experimental evaluation suite, and a **Streamlit** web interface.

We follow an **MVP-First** engineering strategy:
1. **MVP (Core Working System):** Audio upload (`MP3`, `WAV`, `MP4`) $\rightarrow$ Preprocessing (`16 kHz` mono) $\rightarrow$ Language Detection (`EN` / `FR`) $\rightarrow$ Whisper Transcription $\rightarrow$ Timestamped `.srt` / `.txt` export in Streamlit + reproducible evaluation benchmarks.
2. **v2 Upgrade (Live Transcription):** Real-time microphone capture + Voice Activity Detection (VAD) phrase segmentation + live streaming captions.

### Research Question
> *How do different local Whisper model capacities (`tiny`, `base`, `small`) and decoding/noise settings trade off English and French transcription accuracy (WER/CER), spoken language classification performance (Accuracy, Precision, Recall, F1-Score, ROC-AUC vs. a custom MFCC baseline), and inference speed (Real-Time Factor)?*

---

## Planned Repository Structure

```text
VOICESCRIBE/
├── docs/
│   └── VoiceScribe_Teammate_Guide.pdf   # Step-by-step architecture, UI, stack & roadmap guide
├── src/
│   ├── audio_utils.py       # FFmpeg loading, 16 kHz resampling, MFCC & Log-Mel inspection
│   ├── lid_classifier.py    # Custom MFCC + Scikit-Learn EN/FR Language ID baseline
│   ├── asr_engine.py        # Local Whisper inference, language probs, beam/temp tuning
│   ├── subtitle_utils.py    # Timestamp formatting and .srt / .txt export
│   └── vad_mic.py           # [v2] Live microphone capture & VAD silence filtering
├── experiments/
│   ├── eda.py               # Dataset descriptive stats (mean, median, mode) & charts
│   ├── evaluate_asr.py      # WER, CER, and RTF benchmarks (tiny vs base vs small)
│   └── evaluate_lid.py      # EN/FR Language ID: Accuracy, Precision, Recall, F1, ROC-AUC
├── app.py                   # Streamlit web interface
├── requirements.txt
├── TIMELINE.md
└── README.md
```

---

## Quick Teammate Setup (Exact Same Stack Versions)

Both teammates use **Python 3.14.x**, **FFmpeg 9.x**, and exact pinned versions in `requirements.txt`.

1. **Install FFmpeg** (if not already installed on Windows):
   ```powershell
   winget install Gyan.FFmpeg
   ```
2. **Clone & run the one-command environment setup:**
   ```powershell
   git clone https://github.com/marwanedev/voicescribe.git
   cd voicescribe
   .\setup_env.ps1
   ```
3. **Verify at any time that all package versions match 100%:**
   ```powershell
   .\.venv\Scripts\python.exe scripts/check_env.py
   ```

---

## Progress

- [x] Phase 0a: Repository initialized & connected to GitHub
- [x] Teammate Guide PDF created (`docs/VoiceScribe_Teammate_Guide.pdf`)
- [x] Phase 0b: Python `.venv`, pinned `requirements.txt`, and `scripts/check_env.py` verified
- [ ] Phase 1: Literature Review (20–30 papers) — *saved for report writing*
- [x] Phase 2: English & French Dataset (`N=200`) + Exploratory Data Analysis (`experiments/eda.py`)
- [x] Phase 3: MVP Core Pipeline (`src/` + `app.py` for file upload & `.srt`/`.txt`)
- [x] Phase 4: v2 Live Microphone Transcription + VAD (`src/vad_mic.py`)
- [x] Phase 5: Experimental Benchmarks & Ablation Studies (`experiments/evaluate_lid.py` & `experiments/evaluate_asr.py`)
- [ ] Phase 6: Academic Report (Turnitin AI $\le 40\%$), Slides & Defense Prep
