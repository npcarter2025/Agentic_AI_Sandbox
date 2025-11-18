#!/bin/bash
# Environment setup script for Speech_to_LLM scripts
# Usage: source setup_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Transformers cache directory (for model storage)
# Set to a local directory to avoid downloading models to default cache
export TRANSFORMERS_CACHE="${SCRIPT_DIR}/.transformers_cache"
export HF_HOME="${SCRIPT_DIR}/.huggingface"

# Vosk model directory (optional - Vosk will use default if not set)
# export VOSK_MODEL_PATH="${SCRIPT_DIR}/models/vosk"

# Audio device settings (optional - uncomment and set if needed)
# export SOUNDDEVICE_DEVICE=0  # Set to your preferred audio device index

# Python path - add current directory to Python path
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# CUDA/GPU settings (uncomment if using GPU)
# export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Model download settings
export HF_ENDPOINT="https://huggingface.co"  # Default HuggingFace endpoint

# Create cache directories if they don't exist
mkdir -p "${TRANSFORMERS_CACHE}"
mkdir -p "${HF_HOME}"

echo "✅ Environment variables set for Speech_to_LLM"
echo "   TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"
echo "   HF_HOME: ${HF_HOME}"
echo "   PYTHONPATH: ${PYTHONPATH}"

