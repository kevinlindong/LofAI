# lofAI

An AI generated lofi music player that plays infinitely long tracks based on a selected mood.

## Screenshot:

<img width="882" alt="Summary" src="https://github.com/kevinlindong/lofAI/blob/main/lofAI.png">

## How it Works:

- Backend generates lo-fi tracks using Meta's MusicGen text-to-audio transformer model
- Frontend plays tracks sequentially with seamless transitions
- Customize mood and instruments

## Languages and Frameworks:

- **Backend**: Python, FastAPI, WebSockets
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **AI**: Meta's MusicGen Text-To-Speech Transformer Model

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
