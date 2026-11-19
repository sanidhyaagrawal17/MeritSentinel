param(
    [switch]$PullModels,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ResumeProof Portable Setup Script (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check Python version (requires 3.11+)
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pyVersionOutput = & python --version 2>&1
    if ($LASTEXITCODE -ne 0 -and -not $pyVersionOutput) {
        throw "Python not found on PATH."
    }
    Write-Host "Found: $pyVersionOutput" -ForegroundColor Green
    
    # Simple version check >= 3.11
    $versionMatch = [regex]::Match($pyVersionOutput, "Python (\d+)\.(\d+)")
    if ($versionMatch.Success) {
        $major = [int]$versionMatch.Groups[1].Value
        $minor = [int]$versionMatch.Groups[2].Value
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
            Write-Warning "Python 3.11+ is recommended. Found $major.$minor."
        }
    }
} catch {
    Write-Error "Python 3.11+ is required but not found on PATH. Please install Python and try again."
    exit 1
}

# 2. Create virtual environment if missing
Write-Host "[2/5] Setting up virtual environment (.venv)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "Created virtual environment at .venv" -ForegroundColor Green
} else {
    Write-Host "Virtual environment .venv already exists." -ForegroundColor Green
}

# 3. Upgrade pip and install package with dev dependencies
Write-Host "[3/5] Installing package and dependencies (editable mode + dev)..." -ForegroundColor Yellow
$pipPath = ".\.venv\Scripts\python.exe"
& $pipPath -m pip install --upgrade pip
& $pipPath -m pip install -e ".[dev]"
Write-Host "Dependencies installed successfully." -ForegroundColor Green

# 4. Optional Ollama model pulling
if ($PullModels) {
    Write-Host "[4/5] Pulling Ollama models (-PullModels requested)..." -ForegroundColor Yellow
    if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
        Write-Host "Pulling qwen2.5:3b..." -ForegroundColor Cyan
        ollama pull qwen2.5:3b
        Write-Host "Pulling qwen2.5:7b..." -ForegroundColor Cyan
        ollama pull qwen2.5:7b
        Write-Host "Ollama models pulled successfully." -ForegroundColor Green
    } else {
        Write-Warning "Ollama command not found on PATH. Skipping model pull."
    }
} else {
    Write-Host "[4/5] Skipping Ollama model pull (pass -PullModels to pull qwen2.5:3b and qwen2.5:7b)." -ForegroundColor DarkGray
}

# 5. Run tests
if (-not $SkipTests) {
    Write-Host "[5/5] Running pytest verification..." -ForegroundColor Yellow
    & $pipPath -m pytest -q
    Write-Host "Test suite passed successfully!" -ForegroundColor Green
} else {
    Write-Host "[5/5] Skipping tests (-SkipTests requested)." -ForegroundColor DarkGray
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Setup Complete Successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the application in Offline Demo Mode (No Ollama needed):" -ForegroundColor Yellow
Write-Host "  `$env:RESUMEPROOF_USE_FAKE_LLM=`"true`"; .\.venv\Scripts\python -m uvicorn resumeproof.ui.app:app --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "To start the application in Live Ollama Mode:" -ForegroundColor Yellow
Write-Host "  `$env:RESUMEPROOF_USE_FAKE_LLM=`"false`"; .\.venv\Scripts\python -m uvicorn resumeproof.ui.app:app --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "Or use the quick runner script:" -ForegroundColor Yellow
Write-Host "  .\run_app.ps1 -Offline" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
