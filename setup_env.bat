#!/bin/bash
# Windows setup script (run in PowerShell or Command Prompt)

echo "Creating Python virtual environment..."
python -m venv plot-generator-env

echo "Activating environment..."
call plot-generator-env\Scripts\activate.bat

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo "To activate the environment in the future, run:"
echo "plot-generator-env\Scripts\activate"
echo ""
echo "To start training, run:"
echo "python train.py"
