#!/bin/bash
VENV_DIR="venv"

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Run Uvicorn with auto-reload enabled
pip install -r requirements.txt