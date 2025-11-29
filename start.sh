#!/bin/bash

# lofAI Startup Script
# Starts both backend and frontend servers

echo "Starting lofAI Application..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found!"
    echo "Creating .env file template..."
    echo "FAL_KEY=your_fal_api_key_here" > .env
    echo "Please add your fal.ai API key to the .env file and run this script again."
    exit 1
fi

# Check if FAL_KEY is set properly
if grep -q "your_fal_api_key_here" .env || ! grep -q "FAL_KEY=" .env; then
    echo "Warning: FAL_KEY not properly set in .env"
    echo "Make sure to add your fal.ai API key to the .env file"
fi

# Kill any existing processes
echo "Cleaning up any existing processes..."
pkill -f "uvicorn server:app" 2>/dev/null
pkill -f "next dev" 2>/dev/null
sleep 2

# Start backend server in background
echo "Starting backend server..."
./start-backend.sh > backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait a moment for backend to initialize
sleep 3

# Start frontend server in background
echo "Starting frontend server..."
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "   Frontend PID: $FRONTEND_PID"

# Save PIDs to file for easy cleanup later
echo $BACKEND_PID > backend.pid
echo $FRONTEND_PID > frontend.pid

echo ""
echo "lofAI is starting up!"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Logs:"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "To stop the servers, run: ./stop.sh"
echo ""

# Follow the logs (Ctrl+C to exit, servers will keep running)
echo "Following logs (press Ctrl+C to exit, servers will continue running)..."
sleep 2
tail -f backend.log frontend.log
