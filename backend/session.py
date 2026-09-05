# per-user generation session

import os
import math
import threading
import time

import numpy as np

import engine as engine_mod
from music_controls import MusicControls
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
STYLE_RAMP_SECONDS = max(
    engine_mod.FRAME_SECONDS, _env_float("MRT_STYLE_RAMP_SECONDS", 1.2)
)

# the finest slice of a chunk that can carry its own style. chunks are rendered
# a couple of seconds at a time because that is what keeps the model pipelined,
# which would otherwise make a ramp a single step - so a ramp splits the chunk
# instead. each extra slice costs one musiccoca tokenize call, about a
# millisecond, and only while a transition is actually running.
STYLE_STEP_FRAMES = max(1, _env_int("MRT_STYLE_STEP_FRAMES", 10))

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

    def __init__(
        self,
        session_id: str,
        mood: str,
        instrument: str,
        *,
        station: str | None = None,
        controls: dict | None = None,
    ):
        self.id = session_id
        self.status = QUEUED
        self.created_at = time.monotonic()
        self.last_seen = self.created_at

        # opaque mrt2 streaming state - this is what makes the stream endless
        # and what we hold onto across a pause
        self.state = None

        # Planning holds this lock across boundary selection and clock/style
        # advancement. Re-entrancy keeps the smaller public helpers usable
        # without letting an event-loop control request split that transaction.
        self._lock = threading.RLock()
        self.controls = MusicControls.initial(
            mood, instrument, station=station, payload=controls
        )
        self.mood = self.controls.mood
        self.instrument = self.controls.instrument
        self.station = self.controls.station
        self._active_prompt = self.controls.prompt()
        self._active_reference = self.controls.reference()
        self._pending_style: tuple[str, str | None] | None = None
        self._planner_controls = self.controls
        self._pending_planner_controls: MusicControls | None = None
        self.melody = MelodyGuide(session_id)
        self.seed = self.melody.seed & 0xFFFFFFFF

        # New variation invalidates a render already in flight. The worker is
        # the only thread that releases the corresponding MLX state.
        self._render_epoch = 0
        self._reset_requested = False

        # style ramp
        self._current: np.ndarray | None = None
        # the prompt `_current` is exactly, or None while it is a blend of two.
        # the engine caches tokenized conditioning under it.
        self._current_key: str | None = None
        self._active_style_key: str | None = None
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

        # Set by the websocket handler while a client is attached. The worker
        # supplies ``(pcm, render_epoch)`` so deferred event-loop delivery can
        # reject audio from a superseded variation.
        self.sink = None
        self.epoch_sink = None
        self.on_status = None

    # --- control (called from the websocket handler) ---

    def request_style(
        self, mood: str, instrument: str, *, station: str | None = None
    ) -> tuple[str, str]:
        payload = {"mood": mood, "instrument": instrument}
        if station is not None:
            payload["station"] = station
        self.request_controls(payload)
        return self.mood, self.instrument

    def request_controls(self, payload: dict) -> dict:
        """Validate and atomically queue listener music controls."""
        with self._lock:
            next_controls = self.controls.update(payload)
            self.controls = next_controls
            self.mood = next_controls.mood
            self.instrument = next_controls.instrument
            self.station = next_controls.station
            next_target = (next_controls.prompt(), next_controls.reference())
            current_target = (self._active_prompt, self._active_reference)
            if next_target != current_target:
                self._pending_style = next_target
            else:
                # Moving a style away and back before the bar cancels that
                # queued timbre instead of landing a stale intermediate value.
                self._pending_style = None

            if next_controls != self._planner_controls:
                # Score-affecting controls are latched as one bar transaction.
                # Rebuilding an intensity-dependent motif or groove halfway
                # through its phrase is as jarring as a mid-bar key change.
                self._pending_planner_controls = next_controls
            else:
                self._pending_planner_controls = None
            return next_controls.payload()

    def control_payload(self) -> dict:
        with self._lock:
            return self.controls.payload()

    def request_variation(self, new_id: str) -> int:
        with self._lock:
            self.id = new_id
            fresh = MelodyGuide(new_id)
            self.seed = fresh.seed & 0xFFFFFFFF
            self._request_reset_locked()
            return self.seed

    def request_transport_reset(self) -> int:
        """Invalidate in-flight work and restart this take on its next render."""
        with self._lock:
            self._request_reset_locked()
            return self._render_epoch

    def _request_reset_locked(self):
        self._render_epoch += 1
        self._reset_requested = True
        # Audio already queued on the client is reset at the matching transport
        # boundary. Make this session due immediately on its next activation.
        self.playhead = time.monotonic()
        self._paused_lead = 0.0

    def prepare_render(self) -> int:
        """Apply a requested reset on the MLX-owning worker thread."""
        with self._lock:
            if self._reset_requested:
                self.state = None
                self.melody = MelodyGuide(self.id)
                self.seed = self.melody.seed & 0xFFFFFFFF
                self.generated_seconds = 0.0
                self.chunks_rendered = 0
                self._active_wall = 0.0
                self._active_since = time.monotonic() if self.status == ACTIVE else None
                self._planner_controls = self.controls
                self._pending_planner_controls = None
                # A new take is a genuine clean boundary.  Preserve the
                # listener's current control values, but do not carry a stale
                # style ramp or a previously queued station into the new model
                # state.  The next plan embeds the requested target outright.
                self._active_prompt = self.controls.prompt()
                self._active_reference = self.controls.reference()
                self._pending_style = None
                self._current = None
                self._current_key = None
                self._active_style_key = None
                self._ramp_from = None
                self._ramp_to = None
                self._ramp_elapsed = 0.0
                self._ramping = False
                self._reset_requested = False
            return self._render_epoch

    def render_is_current(self, epoch: int) -> bool:
        with self._lock:
            return epoch == self._render_epoch

    def release_state(self):
        """Release MLX-backed recurrent state; called only by the model worker."""
        with self._lock:
            self.state = None

    def deliver_if_current(self, epoch: int, deliver, item) -> bool:
        """Serialize an event-loop delivery against variation resets.

        The worker can validate a render before scheduling a thread-safe
        callback, then lose a race to a variation acknowledgement before that
        callback runs.  Rechecking and queueing while holding the session lock
        makes the old-PCM/ack ordering unambiguous: the PCM is either queued
        before the reset or discarded after it.
        """
        with self._lock:
            if epoch != self._render_epoch:
                return False
            deliver(item)
            return True

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

    @staticmethod
    def _style_key(engine, prompt: str, reference: str | None) -> str:
        make_key = getattr(engine, "style_cache_key", None)
        return make_key(prompt, reference) if make_key is not None else prompt

    @staticmethod
    def _embed_style(engine, prompt: str, reference: str | None) -> np.ndarray:
        return engine.embed(prompt, reference) if reference else engine.embed(prompt)

    def _frames_to_bar(self) -> int:
        composition = self.melody.composition
        step_position = composition.clock.step_position
        steps_per_bar = 16
        position = composition.clock.position(
            composition.active_settings.resolved_bpm
        )
        last_position = composition.clock.last_position
        # A bar begins on the first model frame whose position crossed it, not
        # throughout the entire first sixteenth note. Treating all of step zero
        # as a boundary let changes land up to ~300ms into a bar.
        if last_position is None or last_position.bar < position.bar:
            return 0
        phase = step_position % steps_per_bar
        remaining_steps = steps_per_bar - phase
        bpm = composition.active_settings.resolved_bpm
        return max(
            1,
            math.ceil(
                remaining_steps
                * engine_mod.FRAMES_PER_SECOND
                * 60.0
                / (bpm * 4.0)
            ),
        )

    def control_boundary_frames(self, maximum: int) -> int:
        """Shorten a render so a queued style/score change starts next bar."""
        with self._lock:
            pending = (
                self._pending_style is not None
                or self._pending_planner_controls is not None
            )
        if not pending or self._current is None:
            return maximum
        boundary = self._frames_to_bar()
        return boundary if 0 < boundary < maximum else maximum

    def _activate_pending_style(self, engine):
        with self._lock:
            pending = self._pending_style
            self._pending_style = None
            pending_controls = self._pending_planner_controls
            self._pending_planner_controls = None
            if pending_controls is not None:
                self._planner_controls = pending_controls

        if pending is not None:
            prompt, reference = pending
            self._active_prompt = prompt
            self._active_reference = reference
            self._active_style_key = self._style_key(engine, prompt, reference)
            if self._current is not None:
                # ramp from wherever we are now, which may itself be mid-ramp
                self._ramp_from = self._current
                self._ramp_to = self._embed_style(engine, prompt, reference)
                self._ramp_elapsed = 0.0
                self._ramping = True

    def _style_segments(
        self, engine, frames: int
    ) -> list[tuple[np.ndarray, str | None, int]]:
        """Render style runs without consuming a newly pending target."""

        if self._current is None:
            # first chunk of the session: start on the requested style outright
            self._current = self._embed_style(
                engine, self._active_prompt, self._active_reference
            )
            self._active_style_key = self._style_key(
                engine, self._active_prompt, self._active_reference
            )
            self._current_key = self._active_style_key

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

    def style_plan(self, engine, frames: int) -> list[tuple[np.ndarray, str | None, int]]:
        """Return style runs, landing requested changes on a bar boundary."""
        if frames <= 0:
            return []
        with self._lock:
            has_pending = (
                self._pending_style is not None
                or self._pending_planner_controls is not None
            )

        # The initial station has no previous music to preserve. Subsequent
        # changes wait for the shared composition clock's next bar, avoiding a
        # mid-phrase RVQ jump. The short embedding ramp then starts there.
        if has_pending and self._current is not None:
            wait = self._frames_to_bar()
            if wait > 0:
                return self._style_segments(engine, frames)
            self._activate_pending_style(engine)
            return self._style_segments(engine, frames)

        self._activate_pending_style(engine)
        return self._style_segments(engine, frames)

    def prepare_conditioning(
        self, engine, maximum: int
    ) -> tuple[int, list[engine_mod.ConditioningRun]]:
        """Choose the next musical boundary and advance its plan atomically."""
        with self._lock:
            frames = self.control_boundary_frames(maximum)
            return frames, self._conditioning_plan(engine, frames)

    def conditioning_plan(
        self, engine, frames: int
    ) -> list[engine_mod.ConditioningRun]:
        with self._lock:
            return self._conditioning_plan(engine, frames)

    def _conditioning_plan(
        self, engine, frames: int
    ) -> list[engine_mod.ConditioningRun]:
        """Combine the style surface with one coordinated musical score.

        Harmony, melody, drums, tempo, and form come from one planner. Style
        and score changes retain independent boundaries; splitting only at a
        boundary keeps the per-frame conditioning cache effective.
        """
        if frames <= 0:
            return []
        style_runs = self.style_plan(engine, frames)
        with self._lock:
            controls = self._planner_controls

        self.melody.configure(
            station=controls.station,
            mood=controls.mood,
            bpm=controls.bpm,
            groove=controls.groove,
            intensity=controls.intensity,
            melody_enabled=controls.melody and MELODY_GUIDE_ENABLED,
            drums_enabled=controls.drums,
            guide_mode="guided" if MELODY_GUIDE_ENABLED else "unconstrained",
        )
        note_runs = self.melody.plan_events(controls.mood, frames)
        sampling = self._sampling_for(engine, controls)

        combined: list[engine_mod.ConditioningRun] = []
        style_index = note_index = 0
        style_left = style_runs[0][2]
        note_left = note_runs[0][1]
        while style_index < len(style_runs) and note_index < len(note_runs):
            style, key, _ = style_runs[style_index]
            event, _ = note_runs[note_index]
            take = min(style_left, note_left)
            combined.append(
                engine_mod.ConditioningRun(
                    style=style,
                    key=key,
                    notes=event,
                    drum=event.drum,
                    frames=take,
                    sampling=sampling,
                )
            )
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

        if sum(run.frames for run in combined) != frames:
            raise RuntimeError("conditioning plan lost frames")
        return combined

    @staticmethod
    def _sampling_for(engine, controls: MusicControls) -> engine_mod.SamplingControls:
        base = (
            engine.default_sampling()
            if hasattr(engine, "default_sampling")
            else engine_mod.SamplingControls(1.1, 50, 1.6, 2.4, 4.0)
        )
        # Energy belongs to the score: note density, voicing size, register,
        # bass motion, and comping rhythm. Raising temperature/top-k at the
        # same time made the energetic end less coherent, while the tiny drum
        # CFG adjustment rounded back to the same discrete MRT token anyway.
        return base

    def _ramp_at(self, elapsed: float) -> tuple[np.ndarray, str | None]:
        t = min(1.0, elapsed / STYLE_RAMP_SECONDS)
        if t >= 1.0:
            return self._ramp_to, self._active_style_key
        return _blend(self._ramp_from, self._ramp_to, _smoothstep(t)), None

    def _advance_ramp(self, seconds: float):
        self._ramp_elapsed += seconds
        t = min(1.0, self._ramp_elapsed / STYLE_RAMP_SECONDS)
        if t >= 1.0:
            self._current = self._ramp_to
            self._current_key = self._active_style_key
            self._ramping = False
        else:
            self._current = _blend(self._ramp_from, self._ramp_to, _smoothstep(t))
            self._current_key = None

    def snapshot(self) -> dict:
        controls = self.control_payload()
        return {
            "id": self.id,
            "status": self.status,
            **controls,
            "seed": self.seed,
            "generatedSeconds": round(self.generated_seconds, 1),
            "realtimeFactor": round(self.realtime_factor(), 3),
            "gaps": self.gaps,
        }
