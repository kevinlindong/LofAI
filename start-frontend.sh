#!/bin/bash

# Start frontend script for lofAI

echo "Starting lofAI frontend..."

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start the Next.js development server. exec keeps the tracked process-group
# leader attached to the real service instead of an otherwise idle shell.
echo "Starting frontend server on http://localhost:3000"
exec npm run dev
