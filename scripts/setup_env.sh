#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

if [[ ! -d "${VENV_PATH}" ]]; then
  python3 -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip wheel

# Use the official CPU wheel index to get a build compatible with Apple Silicon (MPS enabled).
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

python -m spacy download en_core_web_sm

echo "Environment setup complete."

