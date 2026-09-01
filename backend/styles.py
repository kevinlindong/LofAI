# style prompts for magenta realtime 2

# MusicCoCa is a style embedding rather than an instruction-following model.
# Keep a controlled template so the mood axis changes mood and the instrument
# axis changes the lead, without silently changing bass, drums, and arrangement
# at the same time.

MOODS = ("somber", "neutral", "lively")
INSTRUMENTS = ("piano", "guitar", "brass")

DEFAULT_MOOD = "neutral"
DEFAULT_INSTRUMENT = "guitar"

MOOD_STYLE = {
    "somber": "intimate melancholy",
    "neutral": "warm mellow",
    "lively": "bright upbeat",
}

INSTRUMENT_STYLE = {
    "piano": "soft felt piano",
    "guitar": "clean jazz guitar",
    "brass": "muted jazz trumpet",
}

PROMPTS = {
    (mood, instrument): (
        f"melodic instrumental {MOOD_STYLE[mood]} lo-fi hip hop, "
        f"{INSTRUMENT_STYLE[instrument]} lead, coherent jazz harmony, "
        "steady laid-back groove, clean balanced mix"
    )
    for mood in MOODS
    for instrument in INSTRUMENTS
}


def normalize(mood: str, instrument: str) -> tuple[str, str]:
    # coerce client input to a known mood/instrument pair
    mood = mood if mood in MOODS else DEFAULT_MOOD
    instrument = instrument if instrument in INSTRUMENTS else DEFAULT_INSTRUMENT
    return mood, instrument


def prompt_for(mood: str, instrument: str) -> str:
    # style text for a mood/instrument pair
    return PROMPTS[normalize(mood, instrument)]


def all_prompts() -> list[str]:
    # every prompt, for warming the embedding cache at startup
    return list(PROMPTS.values())
