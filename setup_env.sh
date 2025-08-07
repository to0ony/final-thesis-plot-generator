#!/bin/bash
# Linux/Mac setup script

echo "Creating Python virtual environment..."
python3 -m venv plot-generator-env

echo "Activating environment..."
source plot-generator-env/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo "To activate the environment in the future, run:"
echo "source plot-generator-env/bin/activate"
echo ""
echo "To start training, run:"
echo "python train.py"
