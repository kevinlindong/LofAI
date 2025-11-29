#!/bin/bash

# Start frontend script for lofAI

echo "Starting lofAI frontend..."

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start the Next.js development server
echo "Starting frontend server on http://localhost:3000"
npm run dev
