#!/bin/bash

# Start backend script for lofAI

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/service-lifecycle.sh"
lofai_service_lifecycle_init

MODEL_SIZE="${MRT_MODEL_SIZE:-mrt2_small}"
MAGENTA_ROOT="${MAGENTA_HOME:-$HOME/Documents/Magenta}/magenta-rt-v2"
BACKEND_HOST="${LOFAI_BACKEND_HOST:-127.0.0.1}"

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
    lofai_run_service_command "$PYTHON_BIN" -m venv venv
fi

source venv/bin/activate

# Install/update backend dependencies
echo "Installing backend dependencies..."
if ! python -m pip --version >/dev/null 2>&1; then
    lofai_run_service_command python -m ensurepip --upgrade
fi
lofai_run_service_command python -m pip install -q -r backend/requirements.txt

# Fetch the model assets (a few GB, first run only). Check the files actually
# consumed by mapped MusicCoCa and SpectroStream; an interrupted/older download
# can leave the directories present but omit mapper.tflite or codec weights.
RESOURCES_COMPLETE=1
for RESOURCE in \
    resources/musiccoca/spm.model \
    resources/musiccoca/text_encoder.tflite \
    resources/musiccoca/mapper.tflite \
    resources/musiccoca/audio_preprocessor.tflite \
    resources/musiccoca/music_encoder.tflite \
    resources/musiccoca/pretrained_vector_quantizer.tflite \
    resources/spectrostream/quantizer.safetensors \
    resources/spectrostream/encoder.safetensors \
    resources/spectrostream/decoder.safetensors; do
    if [ ! -f "$MAGENTA_ROOT/$RESOURCE" ]; then
        RESOURCES_COMPLETE=0
        break
    fi
done

if [ "$RESOURCES_COMPLETE" = "0" ]; then
    echo "Downloading shared resources (MusicCoCa + SpectroStream)..."
    lofai_run_service_command mrt models init --source hf
fi

if [ ! -f "$MAGENTA_ROOT/checkpoints/$MODEL_SIZE.safetensors" ]; then
    echo "Downloading $MODEL_SIZE checkpoint..."
    lofai_run_service_command mrt checkpoints download "$MODEL_SIZE" --source hf
fi

# No .mlxfn export here on purpose. The exported-graph path is faster, but
# every graph exported with the tested MLX 0.32.x builds decodes to
# white noise, reproducible through the library's own `mrt mlx generate`, and
# the same version gap stops it importing google's published .mlxfn. The backend
# runs the checkpoint eagerly instead - see backend/engine.py.

# The model takes a few seconds to load, so --reload is off by default; set
# LOFAI_RELOAD=1 if you are editing the server and can wait for it each time.
echo "Starting backend server on http://$BACKEND_HOST:8000"
cd backend
if [ "${LOFAI_RELOAD:-0}" = "1" ]; then
    lofai_run_service_command \
        python -m uvicorn server:app --reload --host "$BACKEND_HOST" --port 8000 \
        --ws-per-message-deflate false
else
    lofai_run_service_command \
        python -m uvicorn server:app --host "$BACKEND_HOST" --port 8000 \
        --ws-per-message-deflate false
fi
