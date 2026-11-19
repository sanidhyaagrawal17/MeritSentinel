param(
    [switch]$Offline,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ResumeProof App Quick Launcher" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if ($Offline) {
    $env:RESUMEPROOF_USE_FAKE_LLM = "true"
    Write-Host "Mode: OFFLINE DEMO (Using Fake LLM client, zero Ollama requirement)" -ForegroundColor Green
} else {
    $env:RESUMEPROOF_USE_FAKE_LLM = "false"
    Write-Host "Mode: LIVE OLLAMA (Connecting to local Ollama instance)" -ForegroundColor Yellow
}

Write-Host "Starting Uvicorn server on http://127.0.0.1:$Port ..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

$pythonPath = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Error "Virtual environment not found at .venv. Please run .\setup.ps1 first."
    exit 1
}

& $pythonPath -m uvicorn resumeproof.ui.app:app --host 127.0.0.1 --port $Port
