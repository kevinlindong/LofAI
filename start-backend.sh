#!/bin/bash

# Start backend script for lofAI

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

MODEL_SIZE="${MRT_MODEL_SIZE:-mrt2_small}"
MAGENTA_ROOT="${MAGENTA_HOME:-$HOME/Documents/Magenta}/magenta-rt-v2"

echo "Starting lofAI backend..."

# Magenta RealTime 2 needs Apple Silicon for real-time streaming
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "Error: Magenta RealTime 2 streams in real time on Apple Silicon only."
    exit 1
fi

# Create the virtual environment if it does not exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    PYTHON_BIN="$(command -v python3.12 || command -v python3.11 || command -v python3)"
    "$PYTHON_BIN" -m venv venv
fi

source venv/bin/activate

# Install/update backend dependencies
echo "Installing backend dependencies..."
pip install -q -r backend/requirements.txt

# Fetch the model assets (a few GB, first run only)
if [ ! -d "$MAGENTA_ROOT/resources/musiccoca" ]; then
    echo "Downloading shared resources (MusicCoCa + SpectroStream)..."
    mrt models init --source hf
fi

if [ ! -f "$MAGENTA_ROOT/checkpoints/$MODEL_SIZE.safetensors" ]; then
    echo "Downloading $MODEL_SIZE checkpoint..."
    mrt checkpoints download "$MODEL_SIZE" --source hf
fi

# No .mlxfn export here on purpose. The exported-graph path is ~40% faster, but
# every graph mlx 0.32.1 (newest on pypi) exports for this model decodes to
# white noise, reproducible through the library's own `mrt mlx generate`, and
# the same version gap stops it importing google's published .mlxfn. The backend
# runs the checkpoint eagerly instead - see backend/engine.py.

# The model takes a few seconds to load, so --reload is off by default; set
# LOFAI_RELOAD=1 if you are editing the server and can wait for it each time.
RELOAD_FLAG=""
if [ "${LOFAI_RELOAD:-0}" = "1" ]; then
    RELOAD_FLAG="--reload"
fi

echo "Starting backend server on http://localhost:8000"
cd backend
exec python -m uvicorn server:app $RELOAD_FLAG --host 0.0.0.0 --port 8000 --ws-per-message-deflate false
