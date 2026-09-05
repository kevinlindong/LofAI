"""Curated, deliberately short MusicCoCa style targets.

MusicCoCa is a contrastive music-style encoder, not an instruction-following
language model. Audible tags work better than asking it to compose, mix, or
master. Named stations therefore use one compact target while ``custom``
combines the two legacy controls without changing the transport.

An optional owned/licensed WAV can anchor each named station in MusicCoCa's
native audio embedding space. Put files named ``<station>.wav`` in
``MRT_STYLE_REFERENCE_DIR``; the engine blends them with the text embedding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path


MOODS = ("somber", "neutral", "lively")
INSTRUMENTS = ("piano", "guitar", "brass")

DEFAULT_MOOD = "neutral"
DEFAULT_INSTRUMENT = "guitar"
DEFAULT_STATION = "dusty-beats"
CUSTOM_STATION = "custom"


@dataclass(frozen=True)
class Station:
    slug: str
    label: str
    prompt: str
    mood: str
    instrument: str
    bpm: int
    groove: float
    intensity: float


STATIONS: dict[str, Station] = {
    "dusty-beats": Station(
        "dusty-beats",
        "Dusty Beats",
        "instrumental mellow lo-fi hip hop, dusty drums, warm jazz guitar, vinyl",
        "neutral",
        "guitar",
        76,
        0.62,
        0.42,
    ),
    "rainy-piano": Station(
        "rainy-piano",
        "Rainy Piano",
        "instrumental ambient lo-fi, intimate felt piano, sparse brushed drums, soft tape warmth",
        "somber",
        "piano",
        68,
        0.40,
        0.28,
    ),
    "jazz-cafe": Station(
        "jazz-cafe",
        "Jazz Cafe",
        "instrumental late-night jazzhop trio, warm clean guitar, upright bass, brushed drums",
        "neutral",
        "guitar",
        82,
        0.70,
        0.52,
    ),
    "sunlit-groove": Station(
        "sunlit-groove",
        "Sunlit Groove",
        "instrumental soulful jazzhop, muted trumpet, Rhodes keys, crisp relaxed drums",
        "lively",
        "brass",
        94,
        0.78,
        0.68,
    ),
}

MOOD_STYLE = {
    "somber": "melancholy",
    "neutral": "warm mellow",
    "lively": "bright upbeat",
}

INSTRUMENT_STYLE = {
    "piano": "felt piano",
    "guitar": "jazz guitar",
    "brass": "muted trumpet",
}

CUSTOM_PROMPTS = {
    (mood, instrument): (
        f"lo-fi hip hop, {MOOD_STYLE[mood]}, {INSTRUMENT_STYLE[instrument]}"
    )
    for mood in MOODS
    for instrument in INSTRUMENTS
}


def normalize(mood: str, instrument: str) -> tuple[str, str]:
    mood = mood if mood in MOODS else DEFAULT_MOOD
    instrument = instrument if instrument in INSTRUMENTS else DEFAULT_INSTRUMENT
    return mood, instrument


def normalize_station(station: str | None) -> str:
    if station == CUSTOM_STATION:
        return CUSTOM_STATION
    return station if station in STATIONS else DEFAULT_STATION


def station_defaults(station: str | None) -> Station:
    return STATIONS.get(normalize_station(station), STATIONS[DEFAULT_STATION])


def prompt_for(
    mood: str,
    instrument: str,
    station: str = CUSTOM_STATION,
) -> str:
    station = normalize_station(station)
    if station != CUSTOM_STATION:
        return STATIONS[station].prompt
    return CUSTOM_PROMPTS[normalize(mood, instrument)]


def audio_reference_for(station: str | None) -> str | None:
    """Return a station's optional local WAV anchor, if configured and safe."""
    root = os.environ.get("MRT_STYLE_REFERENCE_DIR", "").strip()
    station = normalize_station(station)
    if not root or station == CUSTOM_STATION:
        return None
    candidate = Path(root).expanduser() / f"{station}.wav"
    return str(candidate.resolve()) if candidate.is_file() else None


def all_prompts() -> list[str]:
    # Only listener-facing stations belong on the startup path. Legacy custom
    # combinations are embedded lazily if an older client requests one.
    return [station.prompt for station in STATIONS.values()]


def reference_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for station in STATIONS:
        reference = audio_reference_for(station)
        if reference is not None:
            result[STATIONS[station].prompt] = reference
    return result


def public_options() -> dict:
    return {
        "defaultStation": DEFAULT_STATION,
        "stations": [asdict(station) for station in STATIONS.values()],
        "moods": list(MOODS),
        "instruments": list(INSTRUMENTS),
        "limits": {
            "bpm": [60, 110],
            "groove": [0.0, 1.0],
            "intensity": [0.0, 1.0],
        },
    }
