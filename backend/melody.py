"""MRT2 melody compatibility layer backed by the composition planner.

New integrations should call :meth:`MelodyGuide.plan_events`.  It returns
frame-accurate :class:`composition.PianoRollEvent` objects with explicit onset,
sustain, and release tokens.  The older :meth:`MelodyGuide.plan` and
``piano_roll(int)`` surfaces remain available so existing callers continue to
work while they migrate.
"""

from __future__ import annotations

from collections.abc import Sequence

from composition import (
    FRAME_RATE,
    GuideMode,
    KEY_ROOTS,
    MOOD_BPMS,
    PHRASE_STEPS,
    PianoRollEvent,
    PianoRollRun,
    PlannerSettings,
    SCALES,
    STEPS_PER_BEAT,
    CompositionPlanner,
    stable_seed,
)


# Compatibility names retained for code that imported the original helper's
# constants.  Tempo is now mood-sensitive; this is the neutral default.
TEMPO_BPM = MOOD_BPMS["neutral"]
TONICS = tuple(root + 24 for root in KEY_ROOTS)


def _stable_seed(value: str) -> int:
    return stable_seed(value)


class MelodyGuide:
    """One session's shared harmony/melody/section/drum planner."""

    def __init__(self, session_id: str, *, settings: PlannerSettings | None = None):
        self.composition = CompositionPlanner(session_id, settings=settings)
        self.seed = self.composition.seed
        # Legacy callers understand tonic as the melody register rather than
        # the lower harmony root exposed by ``composition.key_root``.
        self.tonic = self.composition.key_root + 24

    @property
    def frame_index(self) -> int:
        return self.composition.frame_index

    @property
    def _active_mood(self) -> str:
        return self.composition.active_settings.mood

    @property
    def _pending_mood(self) -> str:
        return self.composition.settings.mood

    def configure(self, **controls) -> PlannerSettings:
        """Update controls in place, accepting both wire and internal names."""

        # MusicControls and the websocket protocol expose the concise
        # ``melody``/``drums`` names.  Keep the planner's explicit dataclass
        # field names without forcing every integration point to translate.
        if "melody" in controls:
            controls.setdefault("melody_enabled", controls.pop("melody"))
        if "drums" in controls:
            controls.setdefault("drums_enabled", controls.pop("drums"))
        # Instrument selects the audio/text style in Session; it does not alter
        # the symbolic clock.  Accepting it here lets callers pass the complete
        # listener-control payload without a brittle filtering step.
        controls.pop("instrument", None)
        return self.composition.configure(**controls)

    update = configure

    def plan_events(self, mood: str | None, frames: int) -> list[PianoRollRun]:
        """Return ``(PianoRollEvent, frames)`` runs for direct MRT2 input."""

        return self.composition.plan(mood, frames)

    def event_frames(self, mood: str | None, frames: int) -> list[PianoRollEvent]:
        """Return uncompressed events when callers need exact clock metadata."""

        return self.composition.plan_frames(mood, frames)

    def plan(self, mood: str, frames: int) -> list[tuple[int | None, int]]:
        """Legacy monophonic ``(MIDI note or None, frames)`` plan.

        This view deliberately drops chord, drum, onset, sustain, and release
        information.  It keeps the current session integration operational, but
        ``plan_events`` is required to receive the composition overhaul.
        """

        events = self.event_frames(mood, frames)
        runs: list[tuple[int | None, int]] = []
        for event in events:
            note = event.melody_note
            if runs and runs[-1][0] == note:
                previous, length = runs[-1]
                runs[-1] = (previous, length + 1)
            else:
                runs.append((note, 1))
        return runs

    def _phrase(self, mood: str, phrase_index: int) -> tuple[int | None, ...]:
        """Compatibility/debug view of the varied two-bar motif degrees."""

        return self.composition.phrase_degrees(mood, phrase_index)


def piano_roll(
    value: int | PianoRollEvent | Sequence[int] | None,
) -> list[int] | None:
    """Return a validated MRT2 128-pitch piano roll.

    ``PianoRollEvent`` and 128-token sequences preserve exact ``-1/0/1/2``
    events from the new planner.  Passing an integer retains the old Auto-Strum
    behavior (token ``3``) for compatibility.  ``None`` remains fully
    unconstrained and omits the notes input in the current engine.
    """

    if value is None:
        return None
    if isinstance(value, PianoRollEvent):
        return list(value.tokens)
    if isinstance(value, int) and not isinstance(value, bool):
        if not 0 <= value < 128:
            raise ValueError(f"MIDI note out of range: {value}")
        tokens = [-1] * 128
        tokens[value] = 3
        return tokens
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        tokens = list(value)
        if len(tokens) != 128:
            raise ValueError(f"MRT piano roll must have 128 tokens, got {len(tokens)}")
        if any(token not in (-1, 0, 1, 2, 3) for token in tokens):
            raise ValueError("MRT piano-roll tokens must be in -1..3")
        return tokens
    raise TypeError("expected a MIDI note, PianoRollEvent, 128-token sequence, or None")


def drum_intent(value: PianoRollEvent | int | None) -> int | None:
    """Extract/validate the optional one-channel MRT drum control."""

    if value is None:
        return None
    drum = value.drum if isinstance(value, PianoRollEvent) else value
    if drum not in (-1, 0, 1):
        raise ValueError("MRT drum intent must be -1, 0, or 1")
    return int(drum)


__all__ = [
    "CompositionPlanner",
    "FRAME_RATE",
    "GuideMode",
    "MelodyGuide",
    "MOOD_BPMS",
    "PHRASE_STEPS",
    "PianoRollEvent",
    "PianoRollRun",
    "PlannerSettings",
    "SCALES",
    "STEPS_PER_BEAT",
    "TEMPO_BPM",
    "TONICS",
    "drum_intent",
    "piano_roll",
]
