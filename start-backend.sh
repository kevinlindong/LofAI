#!/bin/bash

# Quick start script for lofAI

echo "Starting lofAI..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env and add your fal.ai API key:"
    echo "  cp .env.example .env"
    echo "  nano .env"
    exit 1
fi

# Check if FAL_KEY is set
if ! grep -q "FAL_KEY=fal_" .env 2>/dev/null; then
    echo "Warning: FAL_KEY not properly set in .env"
    echo "Make sure to add your fal.ai API key to the .env file"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update backend dependencies
echo "Installing backend dependencies..."
pip install -q -r backend/requirements.txt

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Start backend server
echo "Starting backend server on http://localhost:8000"
cd backend
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
