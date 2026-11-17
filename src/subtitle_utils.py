"""
Step 3.4 — Timestamped Subtitle (`.srt`) & Plain Transcript (`.txt`) Generator (`src/subtitle_utils.py`)

This module converts Whisper's decoded timestamp segments into:
1. SubRip Subtitle format (`.srt`) with millisecond precision (`HH:MM:SS,mmm --> HH:MM:SS,mmm`),
   compatible with VLC, YouTube, Premiere Pro, and video players.
2. Clean plain-text transcripts (`.txt`), optionally including segment timestamps.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


def format_srt_timestamp(seconds: float) -> str:
    """
    Convert a timestamp in seconds (e.g. 65.432) into the standard SubRip `.srt`
    format: `HH:MM:SS,mmm` (e.g. `00:01:05,432`).

    Math Breakdown:
    - Total milliseconds = round(seconds * 1000)
    - Hours              = total_ms // 3_600_000
    - Minutes            = (total_ms % 3_600_000) // 60_000
    - Seconds            = (total_ms % 60_000) // 1_000
    - Milliseconds       = total_ms % 1_000
    """
    clamped = max(0.0, float(seconds))
    total_ms = int(round(clamped * 1000))

    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1_000
    millis = remainder % 1_000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    """
    Format a list of Whisper segments (`[{'start': ..., 'end': ..., 'text': ...}]`)
    into a complete `.srt` file string.

    Example block:
        1
        00:00:00,000 --> 00:00:02,120
        Hello and welcome to VoiceScribe.
    """
    blocks: List[str] = []
    for index, seg in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(seg["start"])
        end_ts = format_srt_timestamp(seg["end"])
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        blocks.append(f"{index}\n{start_ts} --> {end_ts}\n{text}")

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def segments_to_txt(
    segments: List[Dict[str, Any]], include_timestamps: bool = False
) -> str:
    """
    Format Whisper segments into a readable `.txt` transcript.

    Parameters
    ----------
    segments : list of dict
        Whisper segment list.
    include_timestamps : bool, default=False
        If True, prefixes each line with `[HH:MM:SS,mmm -> HH:MM:SS,mmm]`.
        If False, joins segment texts cleanly into paragraphs/lines.
    """
    lines: List[str] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if include_timestamps:
            s_ts = format_srt_timestamp(seg["start"])
            e_ts = format_srt_timestamp(seg["end"])
            lines.append(f"[{s_ts} -> {e_ts}] {text}")
        else:
            lines.append(text)

    return "\n".join(lines) + ("\n" if lines else "")


def save_transcript_files(
    segments: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    base_name: str,
) -> Tuple[Path, Path]:
    """
    Write both `<base_name>.srt` and `<base_name>.txt` to `output_dir` (UTF-8 encoded).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    srt_path = out_dir / f"{base_name}.srt"
    txt_path = out_dir / f"{base_name}.txt"

    srt_path.write_text(segments_to_srt(segments), encoding="utf-8")
    txt_path.write_text(segments_to_txt(segments, include_timestamps=False), encoding="utf-8")

    return srt_path, txt_path
