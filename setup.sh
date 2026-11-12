#!/bin/bash
# caramel setup
# Requires: an Apple Silicon Mac (MLX runs on the Metal GPU) and Python 3.12.

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

echo "Installing MLX and dependencies..."
uv pip install mlx==0.32.4 mlx-audio==0.6.5 numpy scipy soundfile huggingface_hub

# Download model (config.json + model.safetensors.partNNN; model.py joins the parts)
echo "Downloading model..."
hf download naklecha/caramel-1 --local-dir "$MODEL_DIR"
(cd "$MODEL_DIR" && shasum -a 256 -c --ignore-missing SHA256SUMS)
python model.py "$MODEL_DIR"

echo ""
echo "Done! Run:"
echo "  source $VENV_DIR/bin/activate"
echo "  python generate.py --model $MODEL_DIR --seconds 30 --realtime"
