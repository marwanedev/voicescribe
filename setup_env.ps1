# VoiceScribe — One-Command Teammate Environment Setup (Windows PowerShell)
# Usage: .\setup_env.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> 1. Checking Python..." -ForegroundColor Cyan
python --version

if (-not (Test-Path ".venv")) {
    Write-Host "==> 2. Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
} else {
    Write-Host "==> 2. Virtual environment (.venv) already exists." -ForegroundColor Cyan
}

Write-Host "==> 3. Installing exact pinned packages from requirements.txt..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "==> 4. Running environment version verification..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe scripts/check_env.py
