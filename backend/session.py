# per-user generation session

import os
import threading
import time

import numpy as np

import engine as engine_mod
from melody import MelodyGuide
import styles


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# how long a slider change takes to fully land, in seconds. the stream glides
# to the new style over this window and then sits exactly on it - this is a
# transition between settings, not a permanent blend of two of them.
STYLE_RAMP_SECONDS = _env_float("MRT_STYLE_RAMP_SECONDS", 1.2)

# the finest slice of a chunk that can carry its own style. chunks are rendered
# a couple of seconds at a time because that is what keeps the model pipelined,
# which would otherwise make a ramp a single step - so a ramp splits the chunk
# instead. each extra slice costs one musiccoca tokenize call, about a
# millisecond, and only while a transition is actually running.
STYLE_STEP_FRAMES = _env_int("MRT_STYLE_STEP_FRAMES", 10)

# Text conditioning describes a sound; the native piano-roll gives that sound
# a coherent foreground line. Keep it on by default, with an escape hatch for
# controlled listening comparisons.
MELODY_GUIDE_ENABLED = _env_int("MRT_MELODY_GUIDE", 1) != 0


ACTIVE = "active"
QUEUED = "queued"
SUSPENDED = "suspended"


def _smoothstep(t: float) -> float:
    # ease the ramp so the transition sounds musical rather than mechanical
    return t * t * (3.0 - 2.0 * t)


def _blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    # Match MusicCoCa's native prompt mixer: normalized weights, linear sum,
    # then RVQ. Renormalizing the vector after mixing changes the token path
    # and is not how Magenta's interactive runtime blends prompt surfaces.
    return ((1.0 - t) * a + t * b).astype(np.float32)


class Session:
    # one user's endless stream: its own model state, style, and pacing

    def __init__(self, session_id: str, mood: str, instrument: str):
        self.id = session_id
        self.status = QUEUED
        self.created_at = time.monotonic()
        self.last_seen = self.created_at

        # opaque mrt2 streaming state - this is what makes the stream endless
        # and what we hold onto across a pause
        self.state = None

        self._lock = threading.Lock()
        self.mood, self.instrument = styles.normalize(mood, instrument)
        self._active_prompt = styles.prompt_for(self.mood, self.instrument)
        self._pending_prompt: str | None = None
        self.melody = MelodyGuide(session_id)
        self.seed = self.melody.seed & 0xFFFFFFFF

        # style ramp
        self._current: np.ndarray | None = None
        # the prompt `_current` is exactly, or None while it is a blend of two.
        # the engine caches tokenized conditioning under it.
        self._current_key: str | None = None
        self._ramp_from: np.ndarray | None = None
        self._ramp_to: np.ndarray | None = None
        self._ramp_elapsed = 0.0
        self._ramping = False

        # pacing: monotonic time at which everything generated so far will have
        # finished playing
        self.playhead = 0.0
        self._paused_lead = 0.0
        self._preserve_paused_audio = False
        self.generated_seconds = 0.0

        # gaps the listener actually heard. the client reports these when its
        # reservoir empties - only it knows, since it is the thing playing.
        self.gaps = 0

        # wall time spent active, for the measured real time factor
        self._active_since: float | None = None
        self._active_wall = 0.0

        # how much audio this session has rendered so far, which is how the
        # worker knows to start it in small chunks and grow into big ones
        self.chunks_rendered = 0

        # set by the websocket handler while a client is attached
        self.sink = None
        self.on_status = None

    # --- control (called from the websocket handler) ---

    def request_style(self, mood: str, instrument: str) -> tuple[str, str]:
        # queue a style change for the next generated chunk
        with self._lock:
            next_mood, next_instrument = styles.normalize(mood, instrument)
            if (next_mood, next_instrument) == (self.mood, self.instrument):
                return self.mood, self.instrument
            self.mood, self.instrument = next_mood, next_instrument
            self._pending_prompt = styles.prompt_for(self.mood, self.instrument)
            return self.mood, self.instrument

    def touch(self):
        self.last_seen = time.monotonic()

    # --- pacing (called from the worker) ---

    def needs_audio(self, now: float, lookahead: float) -> bool:
        # true when this session has less than `lookahead` seconds of audio
        # generated but unplayed
        return self.status == ACTIVE and (self.playhead - now) < lookahead

    def due_in(self, now: float, lookahead: float) -> float:
        # seconds until this session wants its next chunk, so the worker can
        # sleep exactly that long instead of waking up to ask
        if self.status != ACTIVE:
            return float("inf")
        return (self.playhead - now) - lookahead

    def start_clock(self):
        now = time.monotonic()
        self.playhead = now + self._paused_lead
        self._paused_lead = 0.0
        self._preserve_paused_audio = False
        self._active_since = now

    def stop_clock(self, preserve_audio: bool = False):
        # bank the wall time so the real time factor survives a pause
        now = time.monotonic()
        self._paused_lead = max(0.0, self.playhead - now) if preserve_audio else 0.0
        self._preserve_paused_audio = preserve_audio
        if self._active_since is not None:
            self._active_wall += now - self._active_since
            self._active_since = None

    def note_generated(self, seconds: float):
        now = time.monotonic()
        if self.status == SUSPENDED and self._preserve_paused_audio:
            # A pause can race the inference call already in flight. Its PCM is
            # retained by the stopped worklet, so keep the matching lead here.
            self._paused_lead += seconds
            self.playhead = now + self._paused_lead
        elif self.playhead < now:
            # generation fell behind the wall clock; rebase so we don't spend
            # forever trying to make up a deficit we cannot make up
            self.playhead = now
        self.playhead += seconds
        self.generated_seconds += seconds
        self.chunks_rendered += 1

    def note_gap(self):
        self.gaps += 1

    def realtime_factor(self) -> float:
        # seconds of audio produced per second of wall clock. below 1.0 means
        # the machine cannot render this model live, and the client's reservoir
        # drains at (1 - factor) per second.
        wall = self._active_wall
        if self._active_since is not None:
            wall += time.monotonic() - self._active_since
        if wall <= 0.0:
            return 0.0
        return self.generated_seconds / wall

    # --- style (called from the worker) ---

    def style_plan(self, engine, frames: int) -> list[tuple[np.ndarray, str | None, int]]:
        # what to condition the next `frames` frames on, as (style, cache key,
        # frames) segments. one segment unless a transition is running.
        with self._lock:
            pending = self._pending_prompt
            self._pending_prompt = None

        if pending is not None:
            self._active_prompt = pending
            if self._current is not None:
                # ramp from wherever we are now, which may itself be mid-ramp
                self._ramp_from = self._current
                self._ramp_to = engine.embed(pending)
                self._ramp_elapsed = 0.0
                self._ramping = True

        if self._current is None:
            # first chunk of the session: start on the requested style outright
            self._current = engine.embed(self._active_prompt)
            self._current_key = self._active_prompt

        plan: list[tuple[np.ndarray, str | None, int]] = []
        remaining = frames
        while remaining > 0:
            if self._ramping:
                ramp_frames_left = max(
                    1,
                    round(
                        (STYLE_RAMP_SECONDS - self._ramp_elapsed)
                        / engine_mod.FRAME_SECONDS
                    ),
                )
                take = min(STYLE_STEP_FRAMES, remaining, ramp_frames_left)
            else:
                take = remaining
            remaining -= take
            if self._ramping:
                seconds = take * engine_mod.FRAME_SECONDS
                # Condition the whole segment at its midpoint rather than its
                # end, avoiding an immediate jump on every control change.
                style, key = self._ramp_at(self._ramp_elapsed + seconds * 0.5)
                plan.append((style, key, take))
                self._advance_ramp(seconds)
            else:
                plan.append((self._current, self._current_key, take))
        return plan

    def conditioning_plan(
        self, engine, frames: int
    ) -> list[tuple[np.ndarray, str | None, int | None, int]]:
        """Combine the smooth style surface with the symbolic melody runs.

        Style and note changes have independent boundaries. Splitting only at
        either boundary keeps conditioning stable inside a run and avoids
        rebuilding a 128-note block for every 40 ms model step.
        """
        if frames <= 0:
            return []
        style_runs = self.style_plan(engine, frames)
        with self._lock:
            mood = self.mood
        note_runs = (
            self.melody.plan(mood, frames)
            if MELODY_GUIDE_ENABLED
            else [(None, frames)]
        )

        combined: list[tuple[np.ndarray, str | None, int | None, int]] = []
        style_index = note_index = 0
        style_left = style_runs[0][2]
        note_left = note_runs[0][1]
        while style_index < len(style_runs) and note_index < len(note_runs):
            style, key, _ = style_runs[style_index]
            note, _ = note_runs[note_index]
            take = min(style_left, note_left)
            combined.append((style, key, note, take))
            style_left -= take
            note_left -= take

            if style_left == 0:
                style_index += 1
                if style_index < len(style_runs):
                    style_left = style_runs[style_index][2]
            if note_left == 0:
                note_index += 1
                if note_index < len(note_runs):
                    note_left = note_runs[note_index][1]

        if sum(run[3] for run in combined) != frames:
            raise RuntimeError("conditioning plan lost frames")
        return combined

    def _ramp_at(self, elapsed: float) -> tuple[np.ndarray, str | None]:
        t = min(1.0, elapsed / STYLE_RAMP_SECONDS)
        if t >= 1.0:
            return self._ramp_to, self._active_prompt
        return _blend(self._ramp_from, self._ramp_to, _smoothstep(t)), None

    def _advance_ramp(self, seconds: float):
        self._ramp_elapsed += seconds
        t = min(1.0, self._ramp_elapsed / STYLE_RAMP_SECONDS)
        if t >= 1.0:
            self._current = self._ramp_to
            self._current_key = self._active_prompt
            self._ramping = False
        else:
            self._current = _blend(self._ramp_from, self._ramp_to, _smoothstep(t))
            self._current_key = None

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "mood": self.mood,
            "instrument": self.instrument,
            "generatedSeconds": round(self.generated_seconds, 1),
            "realtimeFactor": round(self.realtime_factor(), 3),
            "gaps": self.gaps,
        }
