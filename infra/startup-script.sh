#!/bin/bash

# Log everything to a file for debugging
exec > >(tee /var/log/startup-script.log|logger -t startup-script -s 2>/dev/console) 2>&1

echo "--- Starting BhashaSetu Setup Script ---"

# Update package lists
apt-get update

# Install git and python3-venv
apt-get install -y git python3.10-venv ffmpeg

# Clone your application repository
cd /opt
git clone https://github.com/your-username/your-repo-name.git # <-- IMPORTANT: Replace with your repo URL
cd your-repo-name # <-- IMPORTANT: Replace with your repo folder name

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Install IndicTransToolkit
pip install indic-trans-toolkit

# Make sure all users can access the files
chmod -R 755 /opt/your-repo-name # <-- IMPORTANT: Replace with your repo folder name

echo "--- Setup complete. Starting Uvicorn server ---"

# Run the FastAPI application using Uvicorn
# The app will be accessible on the instance's public IP address
uvicorn main:app --host 0.0.0.0 --port 8000