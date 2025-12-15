#!/usr/bin/env bash

set -euo pipefail

MODEL_DIR=${1:-"models/llama"}
FILE_NAME="Phi-3-mini-4k-instruct-q4_0.gguf"
MODEL_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/${FILE_NAME}?download=1"

mkdir -p "${MODEL_DIR}"
TARGET_PATH="${MODEL_DIR}/${FILE_NAME}"

if [[ -f "${TARGET_PATH}" ]];
then
  echo "Model already exists at ${TARGET_PATH}"
  exit 0
fi

echo "Downloading ${FILE_NAME} to ${TARGET_PATH}" >&2
curl -L "${MODEL_URL}" -o "${TARGET_PATH}"
echo "Download complete."

