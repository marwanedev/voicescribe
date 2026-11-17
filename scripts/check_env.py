r"""
VoiceScribe — Environment & Stack Version Checker (Phase 0b)
Run with:
    .\.venv\Scripts\python.exe scripts/check_env.py

Verifies that both teammates have the exact same Python, FFmpeg, and library versions.
"""

import importlib.metadata
import shutil
import subprocess
import sys

EXPECTED_PACKAGES = {
    "torch": "2.15.0",
    "openai-whisper": "20250625",
    "tiktoken": "0.14.0",
    "librosa": "1.0.1",
    "soundfile": "0.14.0",
    "sounddevice": "0.5.6",
    "scipy": "1.18.1",
    "numpy": "2.5.5",
    "numba": "0.68.1",
    "scikit-learn": "1.9.1",
    "jiwer": "4.0.0",
    "pandas": "3.1.1",
    "matplotlib": "3.11.2",
    "streamlit": "1.67.0",
}


def main() -> int:
    print("=" * 64)
    print("  VoiceScribe — Teammate Environment & Version Check")
    print("=" * 64)

    all_ok = True

    # 1. Python version check
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_match = sys.version_info.major == 3 and sys.version_info.minor == 14
    status = "[OK]" if py_match else "[WARN]"
    print(f"{status:<10} Python         : {py_ver} (Expected: 3.14.x)")
    if not py_match:
        all_ok = False

    # 2. FFmpeg check
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            out = subprocess.check_output(
                ["ffmpeg", "-version"], stderr=subprocess.STDOUT, text=True
            )
            first_line = out.splitlines()[0] if out else "Installed"
            print(f"{'[OK]':<10} FFmpeg         : {first_line}")
        except Exception as exc:
            print(f"{'[FAIL]':<10} FFmpeg         : Error running ffmpeg ({exc})")
            all_ok = False
    else:
        print(
            f"{'[FAIL]':<10} FFmpeg         : NOT FOUND on PATH (run: winget install Gyan.FFmpeg)"
        )
        all_ok = False

    print("-" * 64)
    print(f"{'Status':<10} {'Package':<18} {'Installed':<14} {'Expected':<14}")
    print("-" * 64)

    # 3. Package version checks
    for pkg, expected_ver in EXPECTED_PACKAGES.items():
        try:
            installed_ver = importlib.metadata.version(pkg)
            # Strip local build tags like +cpu if present
            clean_ver = installed_ver.split("+")[0]
            if clean_ver == expected_ver:
                print(f"{'[OK]':<10} {pkg:<18} {installed_ver:<14} {expected_ver:<14}")
            else:
                print(
                    f"{'[DIFF]':<10} {pkg:<18} {installed_ver:<14} {expected_ver:<14}"
                )
                all_ok = False
        except importlib.metadata.PackageNotFoundError:
            print(f"{'[MISSING]':<10} {pkg:<18} {'not installed':<14} {expected_ver:<14}")
            all_ok = False

    print("=" * 64)
    if all_ok:
        print("SUCCESS: Your environment matches the VoiceScribe team standard 100%!")
        return 0
    else:
        print("ACTION NEEDED: Run `pip install -r requirements.txt` inside `.venv`.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
