#!/bin/bash
# Running the setup.sh file. Navigate to directory that contains the file. Then run ./setup.sh
# This will creat a venv, install dependancies.
# Then run source venv/bin/activate to activate the venv so that you can run the streamlit application.

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Virtual environment created and dependencies installed!"
echo "To activate the virtual environment, run: source venv/bin/activate" 