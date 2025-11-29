# lofAI

An ai powered lo-fi music player with fun productivity features!

## How it works:

- Backend generates lo-fi tracks using Meta's MusicGen text-to-audio transformer model
- Frontend plays tracks sequentially with seamless transitions
- Customize mood and instruments

## Languages and Frameworks:

- **Backend**: Python, FastAPI, WebSockets
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **AI**: Meta's MusicGen via fal.ai API
- **Audio**: 10-second MP3 tracks generated on-demand

## Setup:

1. Get a fal.ai api key from https://fal.ai/dashboard/keys

2. Create `.env` file in root directory:
```
FAL_KEY=your_api_key_here
```

3. Install dependencies:
```bash
# backend
python -m venv venv
source venv/bin/activate  # on windows: venv\Scripts\activate
pip install -r backend/requirements.txt

# frontend
cd frontend
npm install
```

## Run:

```bash
# Option 1: Use start script (runs both)
./start.sh

# Option 2: Run separately
./start-backend.sh  # Terminal 1
./start-frontend.sh # Terminal 2
```

Backend: http://localhost:8000  
Frontend: http://localhost:3000

## Stop:

```bash
./stop.sh
```
