"""A restrained symbolic melody guide for Magenta RealTime 2.

MRT2 can invent an arrangement from a text style on its own, but text does not
guarantee that the foreground contains a melody.  The model also accepts a
128-pitch piano-roll every 40 ms.  This module supplies that native control
signal: one scale-bound lead note at a time, with rests, cadences, and small
phrase variations.  The model remains responsible for performance, harmony,
sound design, and the rest of the arrangement.
"""

from __future__ import annotations

import hashlib
import random


FRAME_RATE = 25
TEMPO_BPM = 78.0
STEPS_PER_BEAT = 2
PHRASE_STEPS = 32  # four bars of eighth-note slots in 4/4

# Scale degrees include the octave so every generated variation stays tonal.
SCALES = {
    "somber": (0, 2, 3, 5, 7, 8, 10, 12),       # natural minor
    "neutral": (0, 2, 4, 5, 7, 9, 11, 12),      # major
    "lively": (0, 2, 4, 5, 7, 9, 11, 12),       # major
}

# These are deliberately singable rather than busy.  Each item is a scale
# degree or None for a rest; the last bar always resolves to the tonic.
PATTERNS = {
    "somber": (
        0, None, None, 2, 3, None, 2, None,
        5, None, 4, 3, 2, None, 0, None,
        3, None, 4, 5, 4, None, 2, None,
        1, 2, 4, None, 2, 1, 0, None,
    ),
    "neutral": (
        0, None, 1, 2, 4, None, 2, 1,
        5, None, 4, 2, 1, None, 0, None,
        3, None, 4, 5, 4, 2, 1, None,
        1, 2, 4, None, 2, 1, 0, None,
    ),
    "lively": (
        0, 1, 2, None, 4, 2, 1, 2,
        5, 4, 2, None, 1, 2, 4, None,
        4, 5, 6, 5, 4, 2, 1, None,
        1, 2, 4, 2, 1, None, 0, None,
    ),
}

# A short release between slots keeps the guide articulated and leaves MRT2
# room to phrase the line naturally.
GATE_FRACTION = {
    "somber": 0.72,
    "neutral": 0.78,
    "lively": 0.68,
}

TONICS = (57, 60, 62, 65)  # A3, C4, D4, F4


def _stable_seed(value: str) -> int:
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class MelodyGuide:
    """Produce a continuous, deterministic note plan for one session."""

    def __init__(self, session_id: str):
        self.seed = _stable_seed(session_id)
        self.tonic = TONICS[self.seed % len(TONICS)]
        self.frame_index = 0
        self._active_mood = "neutral"
        self._pending_mood = "neutral"
        self._last_step = -1
        self._phrases: dict[tuple[str, int], tuple[int | None, ...]] = {}

    def plan(self, mood: str, frames: int) -> list[tuple[int | None, int]]:
        """Return ``(midi note or None, frames)`` runs for the next frames."""
        if frames <= 0:
            return []

        self._pending_mood = mood if mood in PATTERNS else "neutral"
        runs: list[tuple[int | None, int]] = []
        for _ in range(frames):
            step_position = (
                self.frame_index * TEMPO_BPM * STEPS_PER_BEAT
                / (FRAME_RATE * 60.0)
            )
            step = int(step_position)
            if step != self._last_step:
                # A control change lands on the next eighth-note boundary, so
                # it cannot cut a held guide note in half.
                self._active_mood = self._pending_mood
                self._last_step = step

            phase = step_position - step
            note = self._note_for_step(self._active_mood, step)
            if phase >= GATE_FRACTION[self._active_mood]:
                note = None

            if runs and runs[-1][0] == note:
                old_note, length = runs[-1]
                runs[-1] = (old_note, length + 1)
            else:
                runs.append((note, 1))
            self.frame_index += 1
        return runs

    def _note_for_step(self, mood: str, step: int) -> int | None:
        phrase_index, phrase_step = divmod(step, PHRASE_STEPS)
        degrees = self._phrase(mood, phrase_index)
        degree = degrees[phrase_step]
        if degree is None:
            return None
        return self.tonic + SCALES[mood][degree]

    def _phrase(self, mood: str, phrase_index: int) -> tuple[int | None, ...]:
        key = (mood, phrase_index)
        cached = self._phrases.get(key)
        if cached is not None:
            return cached

        degrees = list(PATTERNS[mood])
        # Keep every fourth phrase as the recognizable theme.  The intervening
        # phrases receive a few seeded neighbouring scale tones, preserving
        # contour and cadence while avoiding an obvious short loop.
        if phrase_index % 4:
            rng = random.Random(self.seed ^ _stable_seed(f"{mood}:{phrase_index}"))
            for index in range(1, PHRASE_STEPS - 4):
                degree = degrees[index]
                if degree is None or index % 8 == 0 or rng.random() >= 0.14:
                    continue
                direction = -1 if rng.random() < 0.5 else 1
                degrees[index] = max(0, min(7, degree + direction))

        result = tuple(degrees)
        self._phrases[key] = result
        return result


def piano_roll(note: int | None) -> list[int] | None:
    """Encode one guide note using MRT2's permissive active-note token.

    Other pitches stay masked instead of being forced off.  That matches the
    official live MIDI path and lets the model build chords and accompaniment
    around the monophonic lead.
    """
    if note is None:
        return None
    if not 0 <= note < 128:
        raise ValueError(f"MIDI note out of range: {note}")
    tokens = [-1] * 128
    tokens[note] = 3  # active; MRT2 may render it as onset or continuation
    return tokens
