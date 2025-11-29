# fastapi server for lofai music player

import os
import sys
import threading
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from music_generator import MusicGenerator
from connection_manager import ConnectionManager

# config
AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

# global state
current_index = 0
music_generator = None
ws_manager = ConnectionManager()
listener_count = 0
listener_lock = threading.Lock()
initial_generation_started = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialize and cleanup resources
    global music_generator, initial_generation_started
    
    fal_api_key = os.environ.get("FAL_KEY")
    if not fal_api_key:
        print("WARNING: FAL_KEY not set. Music generation will not work.")
        print("Please set the FAL_KEY environment variable with your fal.ai API key.")
    else:
        music_generator = MusicGenerator(fal_api_key)
        # generate first batch in background
        if not initial_generation_started:
            initial_generation_started = True
            threading.Thread(target=generate_initial_batch, daemon=True).start()
    
    yield


app = FastAPI(lifespan=lifespan)

# enable cors for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_initial_batch():
    # generate first batch on startup
    if not music_generator:
        return
    
    try:
        print("Generating initial batch of 10 tracks...")
        music_generator.generate_batch(AUDIO_DIR, 0)
        print("Initial batch generation complete")
    except Exception as e:
        print(f"ERROR: Failed to generate initial music: {e}")


def advance_to_next_track():
    # advance to next track
    global current_index
    
    # advance index sequentially (0-9)
    previous_index = current_index
    current_index = (current_index + 1) % 10
    
    print(f"Advanced from track {previous_index} to track {current_index}")
    
    # generate new batch when looping to 0
    if music_generator and current_index == 0:
        threading.Thread(target=generate_next_batch, daemon=True).start()
    
    return current_index


def generate_next_batch():
    # generate all 10 audio clips
    if not music_generator:
        return
    
    try:
        print("Starting generation of all 10 tracks...")
        music_generator.generate_batch(AUDIO_DIR, 0)
        print("Finished generating all 10 tracks")
        sys.stdout.flush()  # force immediate output
    except Exception as e:
        print(f"ERROR: Failed to generate music: {e}")
        sys.stdout.flush()


@app.get("/api/stream")
async def stream_current_audio(track: int = None):
    # stream specified audio track
    track_index = track if track is not None else current_index
    audio_path = AUDIO_DIR / f"{track_index}.mp3"
    
    if not audio_path.exists():
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Audio not yet generated", "path": str(audio_path)}
        )
    
    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Accept-Ranges": "bytes",
        }
    )


@app.post("/api/update-prompt")
async def update_prompt(data: dict):
    # update music generation prompt
    mood = data.get("mood", "neutral")
    instruments = data.get("instruments", "guitar")
    
    if music_generator:
        music_generator.update_prompt(mood, instruments)
    
    return {"status": "ok", "mood": mood, "instruments": instruments}


@app.post("/api/next-track")
async def next_track():
    # advance to next track and return index
    new_index = advance_to_next_track()
    return {"track": new_index}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # handle websocket connections for listener count
    global listener_count
    
    await ws_manager.connect(websocket)
    await ws_manager.broadcast(str(listener_count))
    
    try:
        while True:
            msg = await websocket.receive_text()
            
            if msg == "listening":
                with listener_lock:
                    listener_count += 1
                await ws_manager.broadcast(str(listener_count))
                
            elif msg == "paused":
                with listener_lock:
                    listener_count = max(0, listener_count - 1)
                await ws_manager.broadcast(str(listener_count))
                
    except Exception:
        pass
    finally:
        with listener_lock:
            listener_count = max(0, listener_count - 1)
        ws_manager.disconnect(websocket)
        await ws_manager.broadcast(str(listener_count))


@app.get("/health")
async def health_check():
    # health check endpoint
    return {
        "status": "ok",
        "current_track": current_index,
        "listeners": listener_count,
        "generator_ready": music_generator is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
