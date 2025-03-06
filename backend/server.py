import threading
import os
from time import sleep
import requests
from typing import Optional
import json

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from websocket_connection_manager import WebSocketConnectionManager

# Constants for MusicGen
FAL_URL = "https://110602490-musicgen-large.gateway.alpha.fal.ai"
LOFI_MOODS = [
    "calm and peaceful",
    "relaxed and mellow",
    "dreamy and ambient",
    "cozy and warm",
    "gentle and soothing"
]
BASE_PROMPT = "Create a lofi hip hop track with {mood}. Include {instrument} with a smooth jazzy melody, soft drums, and ambient textures. The music should be perfect for studying or relaxing."

current_index = -1
t = None
ws_connection_manager = WebSocketConnectionManager()
current_mood = "calm and peaceful"
current_instrument = "piano"

class PromptUpdate(BaseModel):
    mood: str
    instruments: str

def get_fal_headers():
    api_key = os.getenv("FAL_KEY")
    if not api_key:
        raise ValueError("FAL_KEY environment variable not set")
    return {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json"
    }

def generate_music_with_fal(mood: str, instrument: str) -> Optional[bytes]:
    prompt = BASE_PROMPT.format(mood=mood, instrument=instrument)
    
    payload = {
        "model_name": "musicgen-large",
        "prompt": prompt,
        "duration": 60,  # 1 minute
        "top_k": 250,
        "top_p": 0.99,
        "temperature": 0.7,
        "continuation": False,
        "model_version": "stereo-large-v2"
    }

    try:
        response = requests.post(
            FAL_URL,
            headers=get_fal_headers(),
            json=payload,
            timeout=120  # Increased timeout for music generation
        )
        response.raise_for_status()
        
        # Extract the audio URL from the response
        result = response.json()
        if 'audio' in result:
            # Download the generated audio
            audio_response = requests.get(result['audio'])
            audio_response.raise_for_status()
            return audio_response.content
            
    except Exception as e:
        print(f"Error generating music: {str(e)}")
        return None

def generate_new_audio():
    global current_index, current_mood, current_instrument

    offset = 0
    if current_index == 0:
        offset = 5
    elif current_index == 5:
        offset = 0
    else:
        return

    print(f"Generating new batch of lofi tracks starting at index {offset}")
    
    # Generate 5 tracks with different moods
    for i in range(5):
        mood = LOFI_MOODS[i % len(LOFI_MOODS)]
        audio_data = generate_music_with_fal(mood, current_instrument)
        
        if audio_data:
            file_path = f"{i + offset}.mp3"
            try:
                with open(file_path, "wb") as f:
                    f.write(audio_data)
                print(f"Successfully generated and saved track {i + offset}")
            except Exception as e:
                print(f"Error saving audio file {file_path}: {str(e)}")
        else:
            print(f"Failed to generate track {i + offset}")
            
    print("Finished generating audio batch")

def advance():
    global current_index, t

    if current_index == 9:
        current_index = 0
    else:
        current_index = current_index + 1
        
    threading.Thread(target=generate_new_audio).start()
    t = threading.Timer(60, advance)
    t.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize by generating first batch of tracks
    threading.Thread(target=generate_new_audio).start()
    advance()
    yield
    if t:
        t.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/current.mp3")
def get_current_audio():
    return FileResponse(f"{current_index}.mp3")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_connection_manager.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()
            if msg in ["listening", "paused"]:
                await ws_connection_manager.broadcast(msg)
    except:
        ws_connection_manager.disconnect(ws)

@app.post("/update-prompt")
async def update_prompt(prompt_update: PromptUpdate):
    global current_mood, current_instrument
    current_mood = prompt_update.mood
    current_instrument = prompt_update.instruments
    print(f"Updated prompt: Mood - {current_mood}, Instrument - {current_instrument}")
    return {"status": "success"}

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")