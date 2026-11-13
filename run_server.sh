#!/usr/bin/env bash
# Script to launch the AI Sign Language Translator Django application

cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -d "../venv" ]; then
    PYTHON_BIN="../venv/bin/python"
elif [ -d "venv" ]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

export MPLCONFIGDIR="/tmp/matplotlib"
export KERAS_HOME="/tmp/keras"

echo "================================================================="
echo "  AI SIGN LANGUAGE TRANSLATION SYSTEM"
echo "  Production Hybrid TCN + Transformer + Multi-Class SVM"
echo "  Recognition accuracy varies by profile and signing conditions"
echo "================================================================="
echo "Starting Django server on http://127.0.0.1:8000/ ..."
echo "Press Ctrl+C to stop."

$PYTHON_BIN manage.py runserver 0.0.0.0:8000
