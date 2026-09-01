# session lifecycle plus the single generation worker

import logging
import os
import threading
import time
from collections import deque

import engine as engine_mod
import session as session_mod
import styles
from session import ACTIVE, QUEUED, SUSPENDED, Session

log = logging.getLogger(__name__)


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


# audio generated per model call.
#
# this is not a throughput knob - per-call overhead measures flat from 10 to 100
# frames - but it is a pipelining one. 25 frames keeps the GPU pipeline useful
# while bounding startup, control, and transport latency to one second. The old
# 50-frame burst saved only a couple of percentage points but could overshoot
# the reservoir (and delay a style request) by almost two seconds.
CHUNK_FRAMES = _env_int("MRT_CHUNK_FRAMES", 25)
CHUNK_SECONDS = CHUNK_FRAMES / engine_mod.FRAMES_PER_SECOND

# One second exactly meets the fast client's prebuffer in a single call. The
# former 12->24 growth needed two calls to cross that threshold and then risked
# draining the small bank while its much larger third call rendered.
FIRST_CHUNK_FRAMES = _env_int("MRT_FIRST_CHUNK_FRAMES", 25)

# how far ahead of the wall clock a session may run. this is the reservoir the
# client drinks from, and it is also the dominant term in control latency: audio
# already generated cannot be restyled, because the model state has moved past
# it. it used to be six seconds because the model rendered slower than real
# time and the reservoir was all that stood between the listener and a gap.
# The engine adapts above its listener-facing quality floor; the client grows
# its own prebuffer when measured speed is close to the line. This server-side
# lead mainly covers scheduling jitter and bounds how stale a style change can
# be.
LOOKAHEAD_SECONDS = _env_float("MRT_LOOKAHEAD_SECONDS", 2.0)

# concurrent streams. one model instance serves all of them from one thread, so
# this is bounded by how much faster than real time the model runs. watch
# realtimeFactor in /health: it needs to stay above 1.0 per active session.
MAX_ACTIVE = _env_int("MRT_MAX_SESSIONS", 1)

# how long a paused or disconnected session keeps its state before being reaped
SESSION_TTL = _env_float("MRT_SESSION_TTL", 300.0)

# ceiling on retained sessions, so suspended state cannot pile up unbounded
MAX_TOTAL = _env_int("MRT_MAX_SESSIONS_TOTAL", 8)

# longest the worker sleeps when nobody needs audio
IDLE_WAIT = 0.25


class SessionManager:
    # owns every session and the one thread allowed to touch the model

    def __init__(self):
        self.engine = engine_mod.MRTEngine()
        self._sessions: dict[str, Session] = {}
        self._active: list[Session] = []
        self._waiting: deque[Session] = deque()
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._running = False
        self._worker: threading.Thread | None = None
        self._cursor = 0
        self._last_reap = time.monotonic()
        self._last_quality = 0
        # Model/config mutations must stay on the MLX worker thread. The event
        # loop only raises these coalesced feedback flags and wakes it.
        self._pending_gap = False
        self._pending_pressure = False

    # --- startup / shutdown ---

    def start(self):
        # the worker loads the model itself, off the request path, so the server
        # comes up instantly
        with self._lifecycle_lock:
            with self._lock:
                if self._worker is not None and self._worker.is_alive():
                    return
                self._stopping.clear()
                self._wake.clear()
                self._running = True
                prepare = getattr(self.engine, "prepare_start", None)
                if prepare is not None:
                    prepare()
                self._worker = threading.Thread(
                    target=self._run,
                    name="mrt-worker",
                    daemon=False,
                )
                self._worker.start()

    def _load(self) -> bool:
        # must run on the worker thread - see the note in MRTEngine
        started = time.monotonic()
        try:
            self.engine.load()
            if self._stopping.is_set():
                return False
            self.engine.warm_embeddings(styles.all_prompts())
            if self._stopping.is_set():
                return False
            # measure this machine before anyone listens to it, so the first
            # stream already runs at a quality it can actually sustain
            self.engine.calibrate()
            if self._stopping.is_set():
                return False
        except Exception as exc:  # noqa: BLE001 - surfaced to clients via /health
            if self._stopping.is_set():
                return False
            self.engine.load_error = str(exc)
            log.exception("model failed to load")
            self._broadcast_status()
            return False

        log.info("model ready in %.1fs", time.monotonic() - started)
        self._last_quality = self.engine.codebooks
        with self._lock:
            self._promote()
        self._broadcast_status()
        return True

    def stop(self):
        # Detach delivery first. A model call already in flight may need a
        # moment to drain, but none of its result may reach a closing loop.
        with self._lifecycle_lock:
            with self._lock:
                self._running = False
                self._stopping.set()
                worker = self._worker
                for session in self._sessions.values():
                    session.sink = None
                    session.on_status = None

            request_stop = getattr(self.engine, "request_stop", None)
            if request_stop is not None:
                request_stop()
            self._wake.set()
            if worker is threading.current_thread():
                raise RuntimeError("SessionManager.stop() cannot join its own worker")

            if worker is not None:
                # Returning with a live inference thread is not a completed
                # shutdown. The shell supervisor supplies the hard deadline
                # for a genuinely wedged native call.
                worker.join()
                if worker.is_alive():
                    raise RuntimeError("model worker is still running after shutdown")

            with self._lock:
                if self._worker is worker:
                    self._worker = None
                # This is normally already empty from the worker's finally.
                # It also covers stop-before-start and a late failed-load edge.
                self._clear_session_registry_locked()

    # --- session lifecycle (called from the event loop) ---

    def attach(
        self,
        session_id: str | None,
        mood: str,
        instrument: str,
        sink=None,
        on_status=None,
    ) -> tuple[Session, bool]:
        # resume a suspended session by id, or open a new one
        with self._lock:
            if not self._running or self._stopping.is_set():
                raise RuntimeError("session manager is not running")
            existing = self._sessions.get(session_id) if session_id else None
            if existing is not None and existing.sink is not None:
                # someone is already listening on that id (a duplicated tab, or
                # a stale id someone pasted) - give this client its own stream
                # rather than stealing the audio out from under them, and a new
                # id with it so the registry entry is not clobbered below
                existing = None
                session_id = None

            if existing is not None:
                existing.touch()
                existing.sink = sink
                existing.on_status = on_status
                existing.request_style(mood, instrument)
                if existing.status == SUSPENDED:
                    self._enqueue(existing)
                self._promote()
                return existing, True

            session = Session(session_id or self._new_id(), mood, instrument)
            # Bind delivery before promotion so even an unusually fast first
            # inference cannot finish into a missing sink.
            session.sink = sink
            session.on_status = on_status
            self._sessions[session.id] = session
            self._evict_stale()
            self._enqueue(session)
            self._promote()
            return session, False

    def _new_id(self) -> str:
        import uuid

        return uuid.uuid4().hex

    def _enqueue(self, session: Session):
        session.status = QUEUED
        if session not in self._waiting:
            self._waiting.append(session)

    def suspend(self, session: Session, preserve_audio: bool = False):
        # keep the model state, stop generating
        with self._lock:
            if not self._running or self._stopping.is_set():
                session.sink = None
                session.on_status = None
                return
            session.touch()
            if session.status == ACTIVE:
                session.stop_clock(preserve_audio=preserve_audio)
                self._active = [s for s in self._active if s is not session]
            elif session in self._waiting:
                self._waiting.remove(session)
            elif session.status == SUSPENDED and not preserve_audio:
                # A socket may disconnect after an explicit pause. Its worklet
                # reservoir is now gone, so do not preserve the earlier lead.
                session.stop_clock(preserve_audio=False)
            session.status = SUSPENDED
            self._promote()
        self._broadcast_status()

    def resume(self, session: Session):
        with self._lock:
            if not self._running or self._stopping.is_set():
                return
            session.touch()
            if session.status == SUSPENDED:
                self._enqueue(session)
            self._promote()
        self._broadcast_status()

    def report_gap(self, session: Session):
        with self._lock:
            if not self._running or self._stopping.is_set():
                return
            session.note_gap()
            self._pending_gap = True
        self._wake.set()

    def report_pressure(self):
        with self._lock:
            if not self._running or self._stopping.is_set():
                return
            self._pending_pressure = True
        self._wake.set()

    def _promote(self):
        # fill free slots from the waiting queue. caller holds the lock.
        if self._stopping.is_set() or not self.engine.ready:
            return

        changed = False
        while self._waiting and len(self._active) < MAX_ACTIVE:
            session = self._waiting.popleft()
            session.status = ACTIVE
            session.start_clock()
            self._active.append(session)
            changed = True

        if changed:
            self._wake.set()

    def _evict_stale(self):
        # drop suspended sessions past their ttl, then the oldest if still over
        # the retention ceiling. caller holds the lock.
        now = time.monotonic()
        for session in list(self._sessions.values()):
            if (
                session.status == SUSPENDED
                and session.sink is None
                and now - session.last_seen > SESSION_TTL
            ):
                self._forget(session)

        if len(self._sessions) <= MAX_TOTAL:
            return

        suspended = sorted(
            (s for s in self._sessions.values() if s.status == SUSPENDED),
            key=lambda s: s.last_seen,
        )
        while len(self._sessions) > MAX_TOTAL and suspended:
            self._forget(suspended.pop(0))

    def _forget(self, session: Session):
        # caller holds the lock
        self._sessions.pop(session.id, None)
        session.state = None
        session.sink = None
        log.info("reaped session %s", session.id[:8])

    # --- status ---

    def status_for(self, session: Session) -> dict:
        with self._lock:
            payload = {
                "type": "status",
                "state": session.status,
                "listeners": len(self._active),
                "capacity": MAX_ACTIVE,
                # how much faster than real time this machine is rendering. the
                # client sizes its reservoir from it: there is no reason to make
                # someone wait through a deep prebuffer on a box with headroom.
                "realtimeFactor": round(self.engine.realtime_factor(), 3),
            }
            if not self.engine.ready:
                payload["state"] = "loading"
                payload["error"] = self.engine.load_error
            elif session.status == QUEUED:
                try:
                    payload["position"] = list(self._waiting).index(session) + 1
                except ValueError:
                    payload["position"] = 0
            return payload

    def _broadcast_status(self):
        # tell every attached client where it stands
        if self._stopping.is_set():
            return
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            notify = session.on_status
            if notify is not None:
                notify(self.status_for(session))

    def stats(self) -> dict:
        with self._lock:
            return {
                "ready": self.engine.ready,
                "error": self.engine.load_error,
                "model": self.engine.size,
                "backend": self.engine.backend,
                "bits": self.engine.bits,
                "temperature": self.engine.temperature,
                "topK": self.engine.top_k,
                "cfgMusicCoCa": self.engine.cfg_musiccoca,
                "cfgNotes": self.engine.cfg_notes,
                "styleTokenLevels": self.engine.style_token_levels,
                "melodyGuided": session_mod.MELODY_GUIDE_ENABLED,
                "mlxCacheLimitMB": self.engine.mlx_cache_mb,
                "pipelined": self.engine._fast,
                "fastSampler": self.engine._fast_sampling,
                "codebooks": self.engine.codebooks,
                "minCodebooks": self.engine.min_codebooks,
                "maxCodebooks": self.engine.max_codebooks,
                "targetRealtimeFactor": self.engine.target_rtf,
                "renderRealtimeFactor": round(self.engine.realtime_factor(), 3),
                "active": len(self._active),
                "waiting": len(self._waiting),
                "retained": len(self._sessions),
                "capacity": MAX_ACTIVE,
                "chunkSeconds": CHUNK_SECONDS,
                "lookaheadSeconds": LOOKAHEAD_SECONDS,
                "sessions": [s.snapshot() for s in self._sessions.values()],
            }

    # --- the worker ---

    def _run(self):
        try:
            if not self._load():
                return

            log.info(
                "worker up: %d frames/chunk (%.0fms), %.1fs lookahead, %d slots, "
                "%d/%d codebooks",
                CHUNK_FRAMES,
                CHUNK_SECONDS * 1000,
                LOOKAHEAD_SECONDS,
                MAX_ACTIVE,
                self.engine.codebooks,
                self.engine.max_codebooks,
            )

            while not self._stopping.is_set():
                self._apply_feedback()
                session = self._next_due()
                if session is None:
                    self._maybe_reap()
                    # sleep until the earliest session actually wants audio rather
                    # than waking a hundred times a second to find out it does not
                    self._wake.wait(self._idle_wait())
                    self._wake.clear()
                    continue

                frames = self._chunk_frames(session)
                started = time.monotonic()
                try:
                    plan = session.conditioning_plan(self.engine, frames)
                    pcm, next_state = self.engine.generate(
                        session.state, plan, seed=session.seed
                    )
                except Exception as exc:  # noqa: BLE001 - isolate bad sessions
                    if self._stopping.is_set():
                        break
                    log.exception("generation failed for session %s", session.id[:8])
                    notify = session.on_status
                    if notify is not None:
                        notify({"type": "error", "message": str(exc)})
                    self.suspend(session)
                    continue

                # Shutdown may have arrived while native inference was in
                # flight. Drop both its state transition and PCM in that case.
                if self._stopping.is_set():
                    break
                session.state = next_state
                self.engine.note_render(frames, time.monotonic() - started)
                session.note_generated(frames * engine_mod.FRAME_SECONDS)
                with self._lock:
                    sink = None if self._stopping.is_set() else session.sink
                    if sink is not None:
                        sink(pcm)

                if self.engine.codebooks != self._last_quality:
                    # the tuner moved; tell clients so their reservoirs follow
                    self._last_quality = self.engine.codebooks
                    self._broadcast_status()
        finally:
            self._release_worker_resources()

    def _release_worker_resources(self):
        # Session states contain MLX arrays, so release them on the same thread
        # that owns the model before clearing the model itself.
        with self._lock:
            sessions = list(self._sessions.values())
            self._clear_session_registry_locked()
            self._running = False
        for session in sessions:
            session.state = None
            session.sink = None
            session.on_status = None

        close = getattr(self.engine, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001 - cleanup must continue
                log.exception("failed to release inference engine resources")

    def _clear_session_registry_locked(self):
        # Caller holds _lock. Return values are not needed: all retained MLX
        # state is nulled by the worker before engine.close(), or by stop after
        # no worker exists.
        for session in self._sessions.values():
            session.state = None
            session.sink = None
            session.on_status = None
        self._sessions.clear()
        self._active.clear()
        self._waiting.clear()
        self._pending_gap = False
        self._pending_pressure = False
        self._cursor = 0

    def _apply_feedback(self):
        # Called only on the worker, preserving MLX's thread affinity.
        with self._lock:
            gap = self._pending_gap
            pressure = self._pending_pressure
            self._pending_gap = False
            self._pending_pressure = False
        if gap:
            self.engine.note_gap()
        elif pressure:
            # A gap already applies the stronger signal; do not spend two
            # codebooks when both reports race into the same loop iteration.
            self.engine.note_pressure()

    def _chunk_frames(self, session: Session) -> int:
        # start small and double into full chunks: the reservoir is empty at
        # the top of a session, and a listener would otherwise wait a whole
        # chunk for a sound that a tenth of one could have started
        if session.chunks_rendered >= 3:
            return CHUNK_FRAMES
        return min(CHUNK_FRAMES, FIRST_CHUNK_FRAMES << session.chunks_rendered)

    def _next_due(self) -> Session | None:
        # round robin so no session starves when the model is at capacity
        now = time.monotonic()
        with self._lock:
            count = len(self._active)
            for offset in range(count):
                index = (self._cursor + offset) % count
                candidate = self._active[index]
                if candidate.needs_audio(now, LOOKAHEAD_SECONDS):
                    self._cursor = (index + 1) % count
                    return candidate
        return None

    def _idle_wait(self) -> float:
        now = time.monotonic()
        with self._lock:
            if not self._active:
                return IDLE_WAIT
            soonest = min(s.due_in(now, LOOKAHEAD_SECONDS) for s in self._active)
        return max(0.002, min(IDLE_WAIT, soonest))

    def _maybe_reap(self):
        now = time.monotonic()
        if now - self._last_reap < 10.0:
            return
        self._last_reap = now
        with self._lock:
            self._evict_stale()
