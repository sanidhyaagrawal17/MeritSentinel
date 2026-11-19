#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " ResumeProof Portable Setup Script (Linux/macOS Bash)"
echo "============================================================"

# 1. Check Python 3.11+
echo "[1/5] Checking Python installation..."
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version)
    echo "Found: $PY_VERSION"
else
    echo "Error: python3 is required but not found on PATH." >&2
    exit 1
fi

# 2. Create virtual environment if missing
echo "[2/5] Setting up virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment at .venv"
else
    echo "Virtual environment .venv already exists."
fi

# 3. Upgrade pip and install package with dev dependencies
echo "[3/5] Installing package and dependencies (editable mode + dev)..."
PYTHON_BIN="./.venv/bin/python"
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install -e ".[dev]"
echo "Dependencies installed successfully."

# 4. Optional Ollama model pulling
PULL_MODELS=false
SKIP_TESTS=false

for arg in "$@"; do
    case $arg in
        --pull-models)
            PULL_MODELS=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
    esac
done

if [ "$PULL_MODELS" = true ]; then
    echo "[4/5] Pulling Ollama models (--pull-models requested)..."
    if command -v ollama &>/dev/null; then
        echo "Pulling qwen2.5:3b..."
        ollama pull qwen2.5:3b
        echo "Pulling qwen2.5:7b..."
        ollama pull qwen2.5:7b
        echo "Ollama models pulled successfully."
    else
        echo "Warning: ollama command not found on PATH. Skipping model pull." >&2
    fi
else
    echo "[4/5] Skipping Ollama model pull (pass --pull-models to pull qwen2.5:3b and qwen2.5:7b)."
fi

# 5. Run tests
if [ "$SKIP_TESTS" = false ]; then
    echo "[5/5] Running pytest verification..."
    $PYTHON_BIN -m pytest -q
    echo "Test suite passed successfully!"
else
    echo "[5/5] Skipping tests (--skip-tests requested)."
fi

echo "============================================================"
echo " Setup Complete Successfully!"
echo "============================================================"
echo ""
echo "To start the application in Offline Demo Mode (No Ollama needed):"
echo "  RESUMEPROOF_USE_FAKE_LLM=true ./.venv/bin/uvicorn resumeproof.ui.app:app --port 8000"
echo ""
echo "To start the application in Live Ollama Mode:"
echo "  RESUMEPROOF_USE_FAKE_LLM=false ./.venv/bin/uvicorn resumeproof.ui.app:app --port 8000"
echo "============================================================"
