#!/bin/bash

# lofAI Stop Script
# Stops both backend and frontend servers

echo "Stopping lofAI Application..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Try to kill processes by PID files
if [ -f "backend.pid" ]; then
    BACKEND_PID=$(cat backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo "Stopping backend server (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null
    fi
    rm backend.pid
fi

if [ -f "frontend.pid" ]; then
    FRONTEND_PID=$(cat frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null
    fi
    rm frontend.pid
fi

# Fallback: kill by process name
echo "Cleaning up any remaining processes..."
pkill -f "uvicorn server:app" 2>/dev/null
pkill -f "next dev" 2>/dev/null

sleep 1

echo "lofAI stopped successfully!"
