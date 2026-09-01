"""Regression tests for complete application and inference-worker shutdown."""

import asyncio
import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server  # noqa: E402
from session_manager import SessionManager  # noqa: E402


class LifespanManager:
    def __init__(self):
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1


class BlockingLoadEngine:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stop_requested = threading.Event()
        self.closed = threading.Event()
        self.is_ready = False
        self.codebooks = 1
        self.max_codebooks = 1

    @property
    def ready(self):
        return self.is_ready

    def prepare_start(self):
        self.stop_requested.clear()

    def request_stop(self):
        self.stop_requested.set()

    def load(self):
        self.entered.set()
        self.release.wait()

    def warm_embeddings(self, _prompts):
        pass

    def calibrate(self):
        self.is_ready = True

    def close(self):
        self.is_ready = False
        self.closed.set()


class BlockingGenerateEngine:
    size = "fake"
    backend = "fake"
    bits = 0
    temperature = 1.0
    top_k = 1
    cfg_musiccoca = 0.0
    cfg_notes = 0.0
    style_token_levels = 0
    mlx_cache_mb = 0
    _fast = False
    _fast_sampling = False
    min_codebooks = 1
    max_codebooks = 1
    target_rtf = 1.0
    load_error = None

    def __init__(self):
        self.codebooks = 1
        self.is_ready = False
        self.ready_event = threading.Event()
        self.generate_entered = threading.Event()
        self.release_generate = threading.Event()
        self.stop_requested = threading.Event()
        self.closed = threading.Event()
        self.generate_calls = 0

    @property
    def ready(self):
        return self.is_ready

    def prepare_start(self):
        self.stop_requested.clear()

    def request_stop(self):
        self.stop_requested.set()

    def load(self):
        pass

    def warm_embeddings(self, _prompts):
        pass

    def calibrate(self):
        self.is_ready = True
        self.ready_event.set()

    def embed(self, _prompt):
        return np.zeros(4, dtype=np.float32)

    def generate(self, _state, _plan, seed=None):
        del seed
        self.generate_calls += 1
        self.generate_entered.set()
        self.release_generate.wait()
        return b"pcm that must be dropped", object()

    def note_render(self, _frames, _seconds):
        pass

    def realtime_factor(self):
        return 1.0

    def close(self):
        self.is_ready = False
        self.closed.set()


class ShutdownTests(unittest.TestCase):
    def test_lifespan_stops_manager_when_the_application_raises(self):
        fake = LifespanManager()

        async def exercise():
            with patch.object(server, "manager", fake):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    async with server.lifespan(None):
                        raise RuntimeError("boom")

        asyncio.run(exercise())
        self.assertEqual(fake.starts, 1)
        self.assertEqual(fake.stops, 1)

    def test_stop_does_not_return_while_model_loading_is_alive(self):
        manager = SessionManager()
        engine = BlockingLoadEngine()
        manager.engine = engine
        manager.start()
        self.assertTrue(engine.entered.wait(1), "worker never entered fake load")

        stopped = threading.Event()

        def stop():
            manager.stop()
            stopped.set()

        stopper = threading.Thread(target=stop)
        stopper.start()
        self.assertTrue(engine.stop_requested.wait(1))
        self.assertFalse(stopped.wait(0.1), "stop returned with load still running")

        engine.release.set()
        stopper.join(1)
        self.assertFalse(stopper.is_alive())
        self.assertTrue(stopped.is_set())
        self.assertTrue(engine.closed.is_set())
        self.assertIsNone(manager._worker)

    def test_shutdown_drops_generation_that_was_already_in_flight(self):
        manager = SessionManager()
        engine = BlockingGenerateEngine()
        manager.engine = engine
        manager.start()
        self.assertTrue(engine.ready_event.wait(1), "fake engine never became ready")

        delivered = []
        session, _ = manager.attach(
            None,
            "neutral",
            "guitar",
            sink=delivered.append,
        )
        self.assertTrue(engine.generate_entered.wait(1), "generation never started")

        stopper = threading.Thread(target=manager.stop)
        stopper.start()
        self.assertTrue(engine.stop_requested.wait(1))
        self.assertEqual(delivered, [])
        self.assertTrue(stopper.is_alive())

        engine.release_generate.set()
        stopper.join(1)
        self.assertFalse(stopper.is_alive())
        self.assertEqual(delivered, [])
        self.assertIsNone(session.state)
        self.assertEqual(engine.generate_calls, 1)
        self.assertTrue(engine.closed.is_set())

        time.sleep(0.05)
        self.assertEqual(engine.generate_calls, 1, "processing resumed after stop returned")

    def test_stopped_manager_rejects_late_sessions_and_keeps_no_registry(self):
        manager = SessionManager()
        with self.assertRaisesRegex(RuntimeError, "not running"):
            manager.attach(None, "neutral", "guitar", sink=lambda _pcm: None)

        manager.stop()
        self.assertEqual(manager._sessions, {})
        self.assertEqual(manager._active, [])
        self.assertEqual(list(manager._waiting), [])

    def test_start_cannot_replace_worker_while_stop_is_joining_it(self):
        manager = SessionManager()
        engine = BlockingLoadEngine()
        manager.engine = engine
        manager.start()
        self.assertTrue(engine.entered.wait(1))

        stop_finished = threading.Event()
        start_finished = threading.Event()
        stopper = threading.Thread(target=lambda: (manager.stop(), stop_finished.set()))
        starter = threading.Thread(target=lambda: (manager.start(), start_finished.set()))
        stopper.start()
        self.assertTrue(engine.stop_requested.wait(1))
        starter.start()
        self.assertFalse(start_finished.wait(0.1), "start crossed an active stop transition")

        engine.release.set()
        stopper.join(1)
        starter.join(1)
        self.assertTrue(stop_finished.is_set())
        self.assertTrue(start_finished.is_set())

        # The serialized restart may already have completed its fake load; in
        # either case a final stop must leave no worker behind.
        manager.stop()
        self.assertIsNone(manager._worker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
