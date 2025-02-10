#!/bin/bash

# Define Python version and virtual environment name
# PYTHON_VERSION="3.11.5"  # Change this to your required Python version
VENV_DIR="venv"

# Install pyenv if not installed (for managing Python versions)
# if ! command -v pyenv &> /dev/null; then
#     echo "Installing pyenv..."
#     curl https://pyenv.run | bash
#     export PATH="$HOME/.pyenv/bin:$PATH"
#     eval "$(pyenv init --path)"
#     eval "$(pyenv virtualenv-init -)"
# fi

# Install the specific Python version
# pyenv install -s $PYTHON_VERSION
# pyenv local $PYTHON_VERSION

# Create a virtual environment
python3 -m venv $VENV_DIR

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip and install packages from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment setup complete."
