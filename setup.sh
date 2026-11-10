#!/bin/bash
# caramel setup
# Requires: Python 3.12. Generation runs faster than real time on an NVIDIA GPU or an M3 Pro Mac; training needs NVIDIA GPUs with CUDA 12.8+.

set -e

MODEL_DIR="${1:-./caramel-1}"
VENV_DIR="${2:-./.venv}"

echo "=== caramel setup ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create and activate venv
uv venv --python 3.12 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install dependencies (Triton only on Linux; elsewhere kernels.py falls back to plain PyTorch)
echo "Installing PyTorch 2.9.1..."
if [ "$(uname)" = "Linux" ]; then
    uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    uv pip install triton
else
    uv pip install torch==2.9.1
fi

echo "Installing other dependencies..."
uv pip install numpy transformers==4.46.3 safetensors soundfile huggingface_hub

# Download model (config.json + model.safetensors.partNNN; model.py joins the parts on first load)
echo "Downloading model..."
hf download naklecha/caramel-1 --local-dir "$MODEL_DIR"
if command -v sha256sum &> /dev/null; then (cd "$MODEL_DIR" && sha256sum -c --ignore-missing SHA256SUMS)
else (cd "$MODEL_DIR" && shasum -a 256 -c --ignore-missing SHA256SUMS); fi

echo ""
echo "Done! Run:"
echo "  source $VENV_DIR/bin/activate"
echo "  python generate.py --model $MODEL_DIR --seconds 30 --realtime"
