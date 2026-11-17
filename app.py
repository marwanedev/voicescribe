r"""
Step 3.5 — VoiceScribe Streamlit Web Application (`app.py`)
Run with:
    .\.venv\Scripts\streamlit.exe run app.py

Connects the UI controls [1]-[6] from the Teammate Guide PDF directly to our core modules:
- `src/audio_utils.py`    : FFmpeg 16 kHz loading, Log-Mel Spectrogram & MFCC visualization
- `src/asr_engine.py`     : Local Whisper model caching, EN/FR Language ID & transcription
- `src/subtitle_utils.py` : SubRip (.srt) and plain text (.txt) formatting & export
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import streamlit as st

from src.asr_engine import load_whisper_model, transcribe_audio
from src.audio_utils import (
    extract_log_mel_spectrogram,
    extract_mfcc_features,
    get_audio_stats,
    load_and_preprocess_audio,
    plot_audio_features,
)
from src.subtitle_utils import segments_to_srt, segments_to_txt

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

st.set_page_config(
    page_title="VoiceScribe — Bilingual ASR & Subtitle Studio",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_cached_whisper_model(model_size: str) -> Any:
    """Cache the loaded PyTorch Whisper model in memory across Streamlit reruns."""
    return load_whisper_model(model_size=model_size, device="cpu")


def render_sidebar() -> Dict[str, Any]:
    """Render Sidebar Controls [1], [2], and [3] and return user settings."""
    st.sidebar.title("🎙️ VoiceScribe Settings")
    st.sidebar.caption("100% Local CPU Execution • English & French")

    st.sidebar.markdown("### [1] Whisper Model Size")
    model_size = st.sidebar.selectbox(
        "Select Transformer Capacity",
        options=["tiny", "base", "small"],
        index=0,
        help=(
            "tiny (39M params): Fastest on CPU (~0.3x RTF)\n"
            "base (74M params): Balanced speed & accuracy\n"
            "small (244M params): Highest accuracy, slower on CPU"
        ),
    )

    st.sidebar.markdown("### [2] Language & Task Mode")
    lang_choice = st.sidebar.radio(
        "Spoken Language",
        options=[
            "Auto-Detect (English / French)",
            "Force English (en)",
            "Force French (fr)",
        ],
        index=0,
    )
    if "Force English" in lang_choice:
        forced_language: Optional[str] = "en"
    elif "Force French" in lang_choice:
        forced_language = "fr"
    else:
        forced_language = None

    task_choice = st.sidebar.selectbox(
        "Task Mode",
        options=[
            "transcribe (Keep Original Language)",
            "translate (Translate French -> English)",
        ],
        index=0,
    )
    task = "translate" if task_choice.startswith("translate") else "transcribe"

    st.sidebar.markdown("### [3] Decoding Hyperparameters")
    beam_size = st.sidebar.slider(
        "Beam Size (1 = Greedy, 5 = Beam Search)",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        help="Greedy (1) picks the top token immediately. Beam Search (>1) explores multiple candidate sequences.",
    )
    temperature = st.sidebar.slider(
        "Sampling Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.2,
        help="0.0 = deterministic decoding (recommended for accurate ASR).",
    )

    st.sidebar.divider()
    st.sidebar.info(
        "**Architecture Pipeline:**\n"
        "1. `FFmpeg` -> 16 kHz Mono PCM\n"
        "2. `80-Bin Log-Mel` & `13 MFCCs`\n"
        "3. `Whisper Encoder-Decoder`\n"
        "4. `Timestamped .SRT / .TXT`"
    )

    return {
        "model_size": model_size,
        "language": forced_language,
        "task": task,
        "beam_size": beam_size,
        "temperature": temperature,
    }


def resolve_input_audio() -> Optional[Path]:
    """Render [4] Audio Input selector (Upload file or pick built-in EN/FR sample)."""
    input_source = st.radio(
        "Choose Audio Source:",
        options=[
            "Upload Audio / Video File (.mp3, .wav, .mp4, .m4a)",
            "Use Built-in Demo Sample (sample_en.wav / sample_fr.wav)",
        ],
        horizontal=True,
    )

    if input_source.startswith("Upload"):
        uploaded = st.file_uploader(
            "Upload spoken English or French audio/video",
            type=["wav", "mp3", "mp4", "m4a", "flac", "ogg"],
        )
        if uploaded is None:
            return None

        # Save uploaded bytes to a temporary file so FFmpeg can decode it
        suffix = Path(uploaded.name).suffix or ".wav"
        temp_dir = Path(tempfile.gettempdir()) / "voicescribe_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded.name
        temp_path.write_bytes(uploaded.getbuffer())
        st.audio(uploaded)
        return temp_path

    # Built-in sample selector
    sample_choice = st.selectbox(
        "Select Built-in Sample:",
        options=[
            "sample_en.wav — English Speech Sample",
            "sample_fr.wav — French Speech Sample",
        ],
    )
    sample_name = "sample_en.wav" if sample_choice.startswith("sample_en") else "sample_fr.wav"
    sample_path = SAMPLES_DIR / sample_name
    if sample_path.exists():
        st.audio(str(sample_path))
        return sample_path
    else:
        st.warning("Sample WAV files not found. Run `python scripts/test_audio_utils.py` first.")
        return None


def main() -> None:
    settings = render_sidebar()

    st.title("🎙️ VoiceScribe: Bilingual Speech-to-Text & Subtitle System")
    st.markdown(
        "**MVP Studio:** Upload an **English** or **French** audio/video file, inspect its "
        "acoustic features (**Log-Mel Spectrogram & MFCCs**), run local **Whisper** transcription, "
        "and export synchronized **`.srt`** and **`.txt`** subtitles."
    )

    tab_mvp, tab_live = st.tabs(
        ["📁 [4] File Transcription & Subtitles (MVP)", "🎤 Live Microphone Capture (v2 — Phase 4)"]
    )

    with tab_mvp:
        audio_path = resolve_input_audio()

        if audio_path is not None:
            run_clicked = st.button(
                f"🚀 Transcribe with Whisper ({settings['model_size'].upper()})",
                type="primary",
                use_container_width=True,
            )

            if run_clicked:
                with st.status("Running VoiceScribe Pipeline on CPU...", expanded=True) as status:
                    st.write("1️⃣ Decoding media to `16 kHz` mono waveform via FFmpeg (`src/audio_utils.py`)...")
                    waveform = load_and_preprocess_audio(audio_path)
                    audio_stats = get_audio_stats(waveform)

                    st.write("2️⃣ Extracting 80-bin Log-Mel Spectrogram & 13 MFCC features...")
                    log_mel = extract_log_mel_spectrogram(waveform, pad_to_30s=False)
                    mfcc_matrix, mfcc_vec = extract_mfcc_features(waveform, n_mfcc=13)

                    st.write(
                        f"3️⃣ Loading local Whisper `{settings['model_size']}` model & decoding speech (`src/asr_engine.py`)..."
                    )
                    model = get_cached_whisper_model(settings["model_size"])
                    result = transcribe_audio(
                        model=model,
                        waveform=waveform,
                        language=settings["language"],
                        beam_size=settings["beam_size"],
                        temperature=settings["temperature"],
                        task=settings["task"],
                    )

                    st.write("4️⃣ Formatting timestamped `.srt` and `.txt` outputs (`src/subtitle_utils.py`)...")
                    srt_content = segments_to_srt(result["segments"])
                    txt_plain = segments_to_txt(result["segments"], include_timestamps=False)
                    txt_timed = segments_to_txt(result["segments"], include_timestamps=True)

                    status.update(
                        label=f"✅ Transcription Complete in {result['inference_time_sec']}s (RTF: {result['rtf']}x)!",
                        state="complete",
                        expanded=False,
                    )

                # Store in session_state so clicking Download buttons does not wipe results
                st.session_state["last_result"] = {
                    "file_stem": audio_path.stem,
                    "audio_stats": audio_stats,
                    "waveform": waveform,
                    "log_mel": log_mel,
                    "mfcc_matrix": mfcc_matrix,
                    "mfcc_vec": mfcc_vec,
                    "asr": result,
                    "srt": srt_content,
                    "txt_plain": txt_plain,
                    "txt_timed": txt_timed,
                }

        # Render [5] & [6] if results exist in session_state
        if "last_result" in st.session_state:
            data = st.session_state["last_result"]
            asr = data["asr"]
            lid = asr["language_info"]

            st.divider()
            st.subheader("📊 [5] Language Identification & Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Detected Language",
                f"{lid['predicted_lang_name']} ({lid['predicted_lang'].upper()})",
            )
            col2.metric(
                "EN vs FR Confidence",
                f"{lid['confidence'] * 100:.2f}%",
            )
            col3.metric(
                "Audio Duration",
                f"{asr['audio_duration_sec']} s",
            )
            col4.metric(
                "CPU Time (RTF)",
                f"{asr['inference_time_sec']} s ({asr['rtf']}x)",
            )

            en_prob = lid["en_vs_fr_probs"].get("en", 0.0) * 100
            fr_prob = lid["en_vs_fr_probs"].get("fr", 0.0) * 100
            st.caption(
                f"**Whisper Encoder Language Probabilities (Normalized EN/FR):** "
                f"🇬🇧 English: `{en_prob:.2f}%`  |  🇫🇷 French: `{fr_prob:.2f}%`  "
                f"•  Decoding: `beam_size={asr['beam_size']}`, `temperature={asr['temperature']}`"
            )

            with st.expander("🔬 Inspect Acoustic Features (Waveform, 80-Bin Log-Mel Spectrogram & 13 MFCCs)"):
                fig = plot_audio_features(
                    waveform=data["waveform"],
                    log_mel=data["log_mel"],
                    mfcc_matrix=data["mfcc_matrix"],
                    title=f"Acoustic Features — {data['file_stem']}",
                )
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"**Extracted 26-D MFCC Feature Vector (13 Means + 13 Stds for Scikit-Learn):** "
                    f"`{data['mfcc_vec'][:6].round(2).tolist()} ...`"
                )

            st.divider()
            st.subheader("📝 [6] Synchronized Transcript & Subtitle Export (`.srt` / `.txt`)")

            col_text, col_srt = st.columns(2)
            with col_text:
                st.markdown("**Full Transcript (`.txt`)**")
                st.text_area(
                    "Decoded Speech Text",
                    value=asr["text"],
                    height=180,
                    label_visibility="collapsed",
                )
                st.markdown("**Timestamped Segments**")
                st.dataframe(
                    asr["segments"],
                    use_container_width=True,
                    hide_index=True,
                )

            with col_srt:
                st.markdown("**Formatted SubRip Subtitles (`.srt`)**")
                st.code(data["srt"], language="text")

            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button(
                    label="⬇️ Download Subtitles (.srt)",
                    data=data["srt"].encode("utf-8"),
                    file_name=f"{data['file_stem']}.srt",
                    mime="text/plain",
                    type="primary",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    label="⬇️ Download Plain Transcript (.txt)",
                    data=data["txt_plain"].encode("utf-8"),
                    file_name=f"{data['file_stem']}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl_col3:
                st.download_button(
                    label="⬇️ Download Timestamped Text (.txt)",
                    data=data["txt_timed"].encode("utf-8"),
                    file_name=f"{data['file_stem']}_timestamped.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    with tab_live:
        st.info(
            "🚧 **v2 Upgrade (Phase 4):** Once you verify the MVP tab on uploaded & sample files, "
            "we will connect `src/vad_mic.py` here for live microphone recording and VAD phrase segmentation!"
        )


if __name__ == "__main__":
    main()
