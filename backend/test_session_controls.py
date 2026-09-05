"""Session control, seed-reset, and stale-resume regression tests."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from session import SUSPENDED, Session  # noqa: E402
import session_manager as manager_mod  # noqa: E402
from session_manager import SessionManager  # noqa: E402


class PlanEngine:
    def embed(self, _prompt, _reference=None):
        return np.zeros(768, dtype=np.float32)

    @staticmethod
    def style_cache_key(prompt, reference=None):
        return f"{prompt}:{reference or ''}"


class ReadyEngine:
    ready = True


class SpeedEngine:
    ready = True
    target_rtf = 1.15

    def __init__(self, factor, confident=True):
        self.factor = factor
        self.confident = confident

    def realtime_factor(self):
        return self.factor

    def throughput_ready(self):
        return self.confident


class SessionControlTests(unittest.TestCase):
    def test_station_and_planner_controls_reach_one_conditioning_plan(self):
        session = Session(
            "controls",
            "neutral",
            "guitar",
            station="rainy-piano",
        )
        session.request_controls(
            {"bpm": 88, "groove": 0.8, "intensity": 0.7, "drums": False}
        )
        plan = session.conditioning_plan(PlanEngine(), 100)

        self.assertEqual(sum(run.frames for run in plan), 100)
        self.assertTrue(all(run.drum == 0 for run in plan))
        self.assertEqual(session.melody.composition.settings.bpm, 88)
        self.assertEqual(session.control_payload()["groove"], 0.8)

    def test_new_variation_changes_seed_and_invalidates_inflight_render(self):
        session = Session("old", "neutral", "guitar")
        epoch = session.prepare_render()
        old_seed = session.seed

        new_seed = session.request_variation("new")

        self.assertNotEqual(new_seed, old_seed)
        self.assertFalse(session.render_is_current(epoch))
        next_epoch = session.prepare_render()
        self.assertTrue(session.render_is_current(next_epoch))
        self.assertEqual(session.melody.seed & 0xFFFFFFFF, new_seed)

    def test_deferred_old_pcm_is_dropped_after_variation_ack(self):
        session = Session("old-pcm", "neutral", "guitar")
        old_epoch = session.prepare_render()
        queued = []

        # This models a worker callback that has been scheduled onto the event
        # loop but has not run. The variation acknowledgement wins the race.
        session.request_variation("fresh-pcm")
        queued.append("variation-ack")
        delivered = session.deliver_if_current(old_epoch, queued.append, b"old")

        self.assertFalse(delivered)
        self.assertEqual(queued, ["variation-ack"])

    def test_style_and_harmony_change_together_at_a_bar_boundary(self):
        session = Session("bar-change", "neutral", "guitar")
        engine = PlanEngine()
        session.conditioning_plan(engine, 10)
        session.request_controls({"mood": "lively", "instrument": "brass"})
        boundary = session.control_boundary_frames(200)

        self.assertGreater(boundary, 0)
        session.conditioning_plan(engine, boundary)
        self.assertEqual(session.melody.composition.settings.mood, "neutral")

        session.conditioning_plan(engine, 1)
        self.assertEqual(session.melody.composition.settings.mood, "lively")

    def test_rhythm_and_arrangement_controls_also_latch_on_the_bar(self):
        session = Session("bar-arrangement", "neutral", "guitar")
        engine = PlanEngine()
        session.conditioning_plan(engine, 10)
        original_bpm = session.melody.composition.active_settings.bpm
        session.request_controls({"bpm": 96, "intensity": 0.8, "groove": 0.82})
        boundary = session.control_boundary_frames(200)

        session.conditioning_plan(engine, boundary)
        self.assertEqual(session.melody.composition.active_settings.bpm, original_bpm)
        session.conditioning_plan(engine, 1)
        active = session.melody.composition.active_settings
        self.assertEqual(active.bpm, 96)
        self.assertEqual(active.intensity, 0.8)
        self.assertEqual(active.groove, 0.82)

    def test_first_sixteenth_is_not_mistaken_for_the_whole_bar_boundary(self):
        session = Session("narrow-boundary", "neutral", "guitar")
        clock = session.melody.composition.clock
        bpm = session.melody.composition.active_settings.resolved_bpm
        clock.step_position = 16.25
        clock.last_position = clock.position(bpm)

        self.assertGreater(session._frames_to_bar(), 1)

    def test_returning_to_active_style_cancels_pending_bar_change(self):
        session = Session("cancel-style", "neutral", "guitar")
        engine = PlanEngine()
        session.conditioning_plan(engine, 10)

        session.request_controls({"mood": "lively", "instrument": "brass"})
        session.request_controls({"mood": "neutral", "instrument": "guitar"})

        self.assertIsNone(session._pending_style)
        self.assertEqual(session.control_boundary_frames(200), 200)
        session.conditioning_plan(engine, 200)
        self.assertEqual(session.melody.composition.settings.mood, "neutral")

    def test_new_variation_starts_directly_on_requested_style(self):
        session = Session(
            "old-style", "neutral", "guitar", station="dusty-beats"
        )
        engine = PlanEngine()
        session.conditioning_plan(engine, 10)
        session.request_controls({"station": "rainy-piano"})

        session.request_variation("new-style")
        session.prepare_render()
        plan = session.conditioning_plan(engine, 1)

        self.assertIsNone(session._pending_style)
        self.assertFalse(session._ramping)
        self.assertIn("intimate felt piano", plan[0].key)
        self.assertEqual(session.melody.composition.settings.station, "rainy-piano")

    def test_missing_stale_id_is_replaced_instead_of_reseeding_it(self):
        manager = SessionManager()
        manager.engine = ReadyEngine()
        manager._running = True
        manager._new_id = lambda: "fresh-id"

        session, resumed = manager.attach("expired-id", "neutral", "guitar")

        self.assertFalse(resumed)
        self.assertEqual(session.id, "fresh-id")
        self.assertNotIn("expired-id", manager._sessions)

    def test_manager_rekeys_live_session_for_a_new_variation(self):
        manager = SessionManager()
        manager.engine = ReadyEngine()
        manager._running = True
        manager._new_id = lambda: "first"
        session, _ = manager.attach(None, "neutral", "guitar")
        manager._new_id = lambda: "second"

        new_id, seed = manager.new_variation(session)

        self.assertEqual(new_id, "second")
        self.assertEqual(seed, session.seed)
        self.assertNotIn("first", manager._sessions)
        self.assertIs(manager._sessions["second"], session)

    def test_detach_clears_only_its_callbacks_and_invalidates_transport(self):
        manager = SessionManager()
        manager.engine = ReadyEngine()
        manager._running = True
        old_audio = lambda _pcm, _epoch: None
        old_status = lambda _payload: None
        session, _ = manager.attach(
            None,
            "neutral",
            "guitar",
            epoch_sink=old_audio,
            on_status=old_status,
        )
        session.state = object()
        epoch = session.prepare_render()

        manager.detach(session, epoch_sink=old_audio, on_status=old_status)

        self.assertIsNone(session.epoch_sink)
        self.assertIsNone(session.on_status)
        self.assertEqual(session.status, SUSPENDED)
        self.assertFalse(session.render_is_current(epoch))
        self.assertIn(session, manager._release_queue)

    def test_old_socket_detach_cannot_suspend_a_new_receiver(self):
        manager = SessionManager()
        manager.engine = ReadyEngine()
        manager._running = True
        old_audio = lambda _pcm, _epoch: None
        old_status = lambda _payload: None
        session, _ = manager.attach(
            None,
            "neutral",
            "guitar",
            epoch_sink=old_audio,
            on_status=old_status,
        )
        new_audio = lambda _pcm, _epoch: None
        new_status = lambda _payload: None
        session.epoch_sink = new_audio
        session.on_status = new_status
        epoch = session.prepare_render()

        detached = manager.detach(
            session, epoch_sink=old_audio, on_status=old_status
        )

        self.assertFalse(detached)
        self.assertIs(session.epoch_sink, new_audio)
        self.assertIs(session.on_status, new_status)
        self.assertTrue(session.render_is_current(epoch))

    def test_connected_paused_session_is_not_evicted_for_admission(self):
        manager = SessionManager()
        manager.engine = ReadyEngine()
        manager._running = True
        connected = Session("connected", "neutral", "guitar")
        connected.status = SUSPENDED
        connected.epoch_sink = lambda _pcm, _epoch: None
        manager._sessions[connected.id] = connected

        with patch.object(manager_mod, "MAX_TOTAL", 1):
            with self.assertRaisesRegex(RuntimeError, "capacity"):
                manager.attach(None, "neutral", "guitar")

        self.assertIs(manager._sessions[connected.id], connected)

    def test_detached_paused_session_makes_room_without_event_thread_release(self):
        manager = SessionManager()
        manager.engine = ReadyEngine()
        manager._running = True
        detached = Session("detached", "neutral", "guitar")
        detached.status = SUSPENDED
        detached.state = object()
        manager._sessions[detached.id] = detached

        with patch.object(manager_mod, "MAX_TOTAL", 1):
            replacement, _ = manager.attach(None, "neutral", "guitar")

        self.assertNotIn(detached.id, manager._sessions)
        self.assertIn(detached, manager._release_queue)
        self.assertIsNotNone(detached.state)
        self.assertIs(manager._sessions[replacement.id], replacement)

    def test_measured_capacity_promotes_and_queues_without_overcommit(self):
        manager = SessionManager()
        manager.engine = SpeedEngine(2.5)
        manager._running = True
        with patch.object(manager_mod, "MAX_ACTIVE", 3):
            first, _ = manager.attach(None, "neutral", "guitar")
            second, _ = manager.attach(None, "neutral", "guitar")
            third, _ = manager.attach(None, "neutral", "guitar")
            self.assertEqual(len(manager._active), 2)
            self.assertEqual(third.status, "queued")

            manager.engine.factor = 3.6
            self.assertTrue(manager._reconcile_capacity())
            self.assertEqual(len(manager._active), 3)

            manager.engine.factor = 1.2
            self.assertTrue(manager._reconcile_capacity())
            self.assertEqual(manager._active, [first])
            self.assertEqual(second.status, "queued")
            self.assertEqual(third.status, "queued")

    def test_unsettled_throughput_never_opens_extra_slots(self):
        manager = SessionManager()
        manager.engine = SpeedEngine(99.0, confident=False)
        manager._running = True
        with patch.object(manager_mod, "MAX_ACTIVE", 3):
            manager.attach(None, "neutral", "guitar")
            second, _ = manager.attach(None, "neutral", "guitar")

        self.assertEqual(len(manager._active), 1)
        self.assertEqual(second.status, "queued")


if __name__ == "__main__":
    unittest.main(verbosity=2)
