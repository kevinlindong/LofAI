# music generation using fal.ai API

import os
import sys
import requests
import fal_client
import threading
from pathlib import Path


class MusicGenerator:
    # handles ai music generation via fal.ai
    
    # prompt templates for different moods and instruments
    PROMPT_TEMPLATES = {
        "somber": {
            "piano": "Create a melancholic lo-fi beat with gentle, somber {instrument} melodies that evoke a contemplative and introspective atmosphere. The track should feature soft percussion and ambient background elements, with the {instrument} taking center stage. Ensure there are no background noises or interruptions, maintaining a continuous and seamless flow throughout the track. The track must have a clear ending: conclude with a definitive gentle fade-out over the final 2 seconds, or end with a soft final {instrument} chord that resolves peacefully. The beat should be deeply relaxing and tranquil, perfect for a reflective and pensive mood.",
            "guitar": "Create an earthy lo-fi beat that evokes a melancholic, grounded atmosphere with gentle {instrument} as the focal point. Incorporate soft percussion, subtle rustling ambient sounds, and mellow acoustic tones to create a somber, reflective mood. The {instrument} should have a warm, introspective quality with gentle strumming or fingerpicking. The track should have a continuous flow with no background noise or interruptions. The track must have a clear ending: conclude with a definitive gentle fade-out over the final 2 seconds, or end with a soft final {instrument} chord that brings closure, maintaining a calm and contemplative ambiance throughout.",
            "brass": "Create a soothing lo-fi beat featuring gentle, melancholic {instrument} tones. The {instrument} should provide a somber, jazzy atmosphere with warm, muted tones that evoke nostalgia and introspection. Support it with subtle, ambient electronic elements and a smooth, relaxed rhythm. Ensure the track is continuous with no background noise or interruptions. The track must have a clear ending: conclude with a definitive gentle fade-out over the final 2 seconds, or end with a soft {instrument} phrase that resolves smoothly, maintaining a reflective and contemplative atmosphere throughout."
        },
        "neutral": {
            "piano": "Create a gentle lo-fi beat with a smooth, mellow {instrument} melody in the background. The {instrument} should have a warm, comforting tone that creates a peaceful atmosphere. Ensure there are no background noises or interruptions, maintaining a continuous and seamless flow throughout the track. The track must have a clear ending: conclude with a definitive gentle fade-out over the final 2 seconds, or end with a soft {instrument} chord that brings peaceful resolution. The beat should be relaxing and tranquil, perfect for a calm and reflective atmosphere.",
            "guitar": "Create a soothing lo-fi beat featuring gentle, melodic {instrument} riffs. The {instrument} should be the focal point, supported by subtle, ambient electronic elements and a smooth, relaxed rhythm. Ensure the track is continuous with no background noise or interruptions. The track must have a clear ending: conclude with a definitive gentle fade-out over the final 2 seconds, or end with a soft final {instrument} chord that provides closure, maintaining a warm and mellow atmosphere throughout.",
            "brass": "Create an ambient lo-fi beat with tranquil {instrument} tones creating an ethereal atmosphere. Use soft, atmospheric pads alongside warm {instrument} melodies, gentle rhythms, and minimalistic percussion to evoke a sense of calm and serenity. Ensure the track is continuous with no background noise or interruptions. The track must have a clear ending: conclude with a definitive gentle fade-out over the final 2 seconds, or end with a soft {instrument} note that resolves peacefully, maintaining a soothing and immersive ambiance throughout."
        },
        "lively": {
            "piano": "Create a futuristic lo-fi beat that blends modern electronic elements with uplifting {instrument} melodies. Incorporate bright, energetic {instrument} tones and lively, rhythmic beats to evoke a sense of optimism and vibrant energy. Ensure the track is continuous with no background noise or interruptions. The track must have a clear ending: conclude with a definitive upbeat fade-out over the final 2 seconds, or end with an energetic final {instrument} flourish or chord progression that brings the track to a satisfying close, maintaining an upbeat and positive atmosphere throughout while adding a touch of contemporary flair.",
            "guitar": "Create a lively lo-fi beat featuring upbeat {instrument} riffs that energize and uplift. The {instrument} should be bright and cheerful, with rhythmic strumming supported by bouncy electronic elements. Ensure there are no background noises or interruptions, maintaining a continuous and seamless flow throughout the track. The track must have a clear ending: conclude with a definitive upbeat fade-out over the final 2 seconds, or end with a bright final {instrument} chord or strum that brings energetic closure. The beat should be vibrant and positive, perfect for an energetic and motivating atmosphere.",
            "brass": "Create an energetic lo-fi beat with lively {instrument} sections that bring a jazzy, upbeat atmosphere. The {instrument} should provide bright, bold tones with bouncy rhythms and syncopated patterns. Support it with dynamic percussion and uplifting electronic elements. Ensure the track is continuous with no background noise or interruptions. The track must have a clear ending: conclude with a definitive upbeat fade-out over the final 2 seconds, or end with a bright {instrument} phrase or jazzy flourish that brings the track to a satisfying conclusion, maintaining a cheerful and vibrant mood throughout."
        }
    }
    
    def __init__(self, api_key: str):
        # initialize music generator with api key
        self.api_key = api_key
        self.current_mood = "neutral"
        self.current_instrument = "guitar"
        self._generation_lock = threading.Lock()  # ensures sequential generation
        
        # Set API key for fal_client
        os.environ["FAL_KEY"] = api_key
    
    def update_prompt(self, mood: str, instrument: str):
        # update mood and instrument preferences
        self.current_mood = mood
        self.current_instrument = instrument
    
    def get_current_prompt(self) -> str:
        # get current prompt based on mood and instrument
        template = self.PROMPT_TEMPLATES.get(self.current_mood, {}).get(
            self.current_instrument,
            self.PROMPT_TEMPLATES["neutral"]["guitar"]
        )
        # replace {instrument} placeholder
        return template.format(instrument=self.current_instrument)
    
    def generate_batch(self, output_dir: Path, offset: int = 0):
        # generate all 10 audio clips using fal.ai
        # ensure only one batch generates at a time
        with self._generation_lock:
            output_dir.mkdir(exist_ok=True)
            
            # generate all 10 clips
            prompt = self.get_current_prompt()
            
            for i in range(10):
                clip_number = i
                output_path = output_dir / f"{clip_number}.mp3"
                temp_path = output_dir / f"{clip_number}.mp3.tmp"
                
                try:
                    print(f"Generating track {clip_number}...")
                    sys.stdout.flush()  # force immediate output
                    
                    # generate to temp file first
                    audio_data = self._generate_single_clip(prompt)
                    
                    # save to temp file
                    with open(temp_path, "wb") as f:
                        f.write(audio_data)
                    
                    # move temp to final (atomic)
                    temp_path.replace(output_path)
                    
                    print(f"Track {clip_number} saved to audio/{clip_number}.mp3")
                    sys.stdout.flush()  # Force output to appear immediately
                    
                except Exception as e:
                    print(f"ERROR: Failed to generate track {clip_number}: {e}")
                    sys.stdout.flush()
                    # cleanup temp file
                    if temp_path.exists():
                        temp_path.unlink()
                    # continue with next clip
    
    def _generate_single_clip(self, prompt: str) -> bytes:
        # generate single audio clip using fal.ai API
        try:
            # call fal.ai api
            result = fal_client.run(
                "fal-ai/musicgen",
                arguments={
                    "prompt": prompt,
                    "model_version": "large",
                    "duration": 10,
                    "output_format": "mp3"
                }
            )
            
            # extract audio url from response
            if not isinstance(result, dict):
                raise Exception("Invalid response format from fal.ai")
            
            # check possible response structures
            audio_url = None
            if 'audio_url' in result:
                audio_url_data = result['audio_url']
                audio_url = audio_url_data.get('url') if isinstance(audio_url_data, dict) else audio_url_data
            elif 'audio_file' in result:
                audio_file = result['audio_file']
                audio_url = audio_file.get('url') if isinstance(audio_file, dict) else audio_file
            elif 'url' in result:
                audio_url = result['url']
            elif 'data' in result:
                data = result['data']
                if isinstance(data, dict):
                    if 'audio_file' in data:
                        af = data['audio_file']
                        audio_url = af.get('url') if isinstance(af, dict) else af
                    elif 'url' in data:
                        audio_url = data['url']
            
            if not audio_url:
                raise Exception(f"No audio URL found. Available keys: {list(result.keys())}")
            
            # download audio file
            audio_response = requests.get(audio_url, timeout=120)
            audio_response.raise_for_status()
            
            return audio_response.content
            
        except Exception as e:
            raise

