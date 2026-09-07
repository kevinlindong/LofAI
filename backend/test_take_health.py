"""Drift detection and repair for takes that grow a hiss bed over time."""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import take_health  # noqa: E402
from take_health import TakeFloorMonitor, crossfade_pcm  # noqa: E402
from session import Session  # noqa: E402

RATE = take_health.SAMPLE_RATE
CHUNK_FRAMES_48K = int(0.4 * RATE)


def _stereo_int16(mono: np.ndarray) -> bytes:
    clipped = np.clip(mono, -1.0, 1.0)
    frame = np.round(clipped * 32767.0).astype(np.int16)
    return np.repeat(frame, 2).tobytes()


def _music_second(rng, gap_floor_dbfs: float, gap_hiss: bool, level_db=-20.0):
    """One second: 0.8s of tone 'music', 0.2s of gap at the given floor."""
    t = np.arange(int(0.8 * RATE)) / RATE
    music = (10 ** (level_db / 20)) * np.sin(2 * np.pi * 220.0 * t)
    gap_len = RATE - music.size
    amplitude = 10 ** (gap_floor_dbfs / 20)
    if gap_hiss:
        gap = rng.normal(0.0, amplitude, gap_len)  # broadband: hiss
    else:
        tg = np.arange(gap_len) / RATE
        gap = amplitude * np.sin(2 * np.pi * 180.0 * tg)  # low rumble: not hiss
    return np.concatenate([music, gap]).astype(np.float64)


def _feed_seconds(monitor, seconds_audio: np.ndarray, *, chunk_frames=CHUNK_FRAMES_48K):
    pcm = _stereo_int16(seconds_audio)
    frame_bytes = 4
    chunk = chunk_frames * frame_bytes
    for start in range(0, len(pcm), chunk):
        monitor.observe(pcm[start : start + chunk])


class TakeFloorMonitorTests(unittest.TestCase):
    def _run_take(self, drift: bool, *, louder_music_instead=False) -> TakeFloorMonitor:
        rng = np.random.default_rng(11)
        monitor = TakeFloorMonitor()
        clean = np.concatenate(
            [_music_second(rng, -50.0, gap_hiss=True) for _ in range(140)]
        )
        _feed_seconds(monitor, clean)
        self.assertFalse(monitor.drifted, "clean take must not report drift")

        if drift:
            tail = np.concatenate(
                [_music_second(rng, -28.0, gap_hiss=True) for _ in range(80)]
            )
        elif louder_music_instead:
            tail = np.concatenate(
                [
                    _music_second(rng, -50.0, gap_hiss=True, level_db=-12.0)
                    for _ in range(80)
                ]
            )
        else:
            tail = np.concatenate(
                [_music_second(rng, -50.0, gap_hiss=True) for _ in range(80)]
            )
        _feed_seconds(monitor, tail)
        return monitor

    def test_sustained_floor_and_high_band_rise_reports_drift(self):
        self.assertTrue(self._run_take(drift=True).drifted)

    def test_stable_take_never_reports_drift(self):
        self.assertFalse(self._run_take(drift=False).drifted)

    def test_denser_louder_music_is_not_drift(self):
        # The performance getting louder moves the loud blocks, not the gaps.
        self.assertFalse(
            self._run_take(drift=False, louder_music_instead=True).drifted
        )

    def test_low_frequency_floor_rise_alone_is_not_hiss(self):
        # A warm low rumble without a high-band rise is not a hiss bed.
        rng = np.random.default_rng(7)
        monitor = TakeFloorMonitor()
        clean = np.concatenate(
            [_music_second(rng, -50.0, gap_hiss=False) for _ in range(140)]
        )
        _feed_seconds(monitor, clean)
        rumble = np.concatenate(
            [_music_second(rng, -30.0, gap_hiss=False) for _ in range(80)]
        )
        _feed_seconds(monitor, rumble)
        self.assertFalse(monitor.drifted)

    def test_inaudible_floors_never_trigger(self):
        rng = np.random.default_rng(5)
        monitor = TakeFloorMonitor()
        clean = np.concatenate(
            [_music_second(rng, -75.0, gap_hiss=True) for _ in range(140)]
        )
        _feed_seconds(monitor, clean)
        quiet_rise = np.concatenate(
            [_music_second(rng, -55.0, gap_hiss=True) for _ in range(80)]
        )
        _feed_seconds(monitor, quiet_rise)
        self.assertFalse(monitor.drifted, "a rise below audibility is not worth a cut")

    def test_observe_tolerates_junk_chunks(self):
        monitor = TakeFloorMonitor()
        monitor.observe(b"")
        monitor.observe(b"pcm")
        monitor.observe(b"\x00" * 7)
        self.assertFalse(monitor.drifted)

    def test_opposite_phase_stereo_hiss_is_still_detected(self):
        rng = np.random.default_rng(11)
        monitor = TakeFloorMonitor()
        for second in range(230):
            audio = _music_second(
                rng, -50.0 if second < 140 else -28.0, gap_hiss=True
            )
            stereo = np.frombuffer(_stereo_int16(audio), dtype="<i2").copy()
            stereo = stereo.reshape(-1, 2)
            # Keep the performance centered; put just its noise bed in the
            # stereo side channel. It is audible but cancels in a mono fold.
            stereo[int(0.8 * RATE) :, 1] *= -1
            monitor.observe(stereo.tobytes())
            if second == 139:
                self.assertFalse(monitor.drifted)
        self.assertTrue(monitor.drifted)

    def test_sustain_measures_audio_time_independent_of_chunk_size(self):
        rng = np.random.default_rng(11)
        short_chunks = TakeFloorMonitor()
        long_chunks = TakeFloorMonitor()
        for second in range(230):
            audio = _music_second(
                rng, -50.0 if second < 140 else -28.0, gap_hiss=True
            )
            _feed_seconds(short_chunks, audio, chunk_frames=int(0.04 * RATE))
            _feed_seconds(long_chunks, audio, chunk_frames=int(0.8 * RATE))
            self.assertEqual(short_chunks.drifted, long_chunks.drifted)
        self.assertTrue(short_chunks.drifted, "sustained hiss must be detected")
        self.assertEqual(
            list(short_chunks._drift_votes), list(long_chunks._drift_votes)
        )

    def test_empty_chunks_do_not_turn_a_brief_rise_into_sustained_drift(self):
        rng = np.random.default_rng(11)
        monitor = TakeFloorMonitor()
        for second in range(200):
            _feed_seconds(
                monitor,
                _music_second(rng, -50.0 if second < 140 else -28.0, gap_hiss=True),
            )
        self.assertTrue(any(monitor._drift_votes), "the current floor has risen")
        self.assertFalse(monitor.drifted, "the rise has not lasted long enough")
        before = list(monitor._drift_votes)
        for _ in range(200):
            monitor.observe(b"")
            monitor.observe(b"pcm")
        self.assertEqual(list(monitor._drift_votes), before)
        self.assertFalse(monitor.drifted)

    def test_sustained_brightening_is_detected_without_overall_floor_rise(self):
        rng = np.random.default_rng(17)
        monitor = TakeFloorMonitor()
        gap_t = np.arange(int(0.2 * RATE)) / RATE
        for second in range(230):
            audio = _music_second(rng, -30.0, gap_hiss=False)
            hiss_db = -60.0 if second < 140 else -36.0
            audio[-gap_t.size :] += rng.normal(
                0.0, 10 ** (hiss_db / 20), gap_t.size
            )
            _feed_seconds(monitor, audio)
        self.assertTrue(monitor.drifted)
        current_floor, _ = take_health._floor_profile(
            monitor._trailing_rms, monitor._trailing_high
        )
        self.assertLess(
            take_health._dbfs(current_floor)
            - take_health._dbfs(monitor._baseline_floor),
            3.0,
            "spectral drift must not need a large overall volume increase",
        )

    def test_reset_forgets_the_baseline(self):
        monitor = self._run_take(drift=True)
        self.assertTrue(monitor.drifted)
        monitor.reset()
        self.assertFalse(monitor.drifted)
        self.assertIn("warming", monitor.describe())


class CrossfadePcmTests(unittest.TestCase):
    def test_equal_power_crossfade_moves_old_to_new(self):
        frames = 19_200
        old = _stereo_int16(np.full(frames, 0.5))
        new = _stereo_int16(np.full(frames, -0.5))
        mixed = np.frombuffer(crossfade_pcm(old, new), dtype=np.int16).reshape(-1, 2)
        self.assertEqual(len(mixed), frames)
        self.assertAlmostEqual(mixed[0, 0] / 32767.0, 0.5, places=2)
        self.assertAlmostEqual(mixed[-1, 0] / 32767.0, -0.5, places=2)
        middle = mixed[frames // 2, 0] / 32767.0
        self.assertLess(abs(middle), 0.05)

    def test_length_mismatch_uses_the_shorter_chunk(self):
        old = _stereo_int16(np.full(1000, 0.25))
        new = _stereo_int16(np.full(800, -0.25))
        mixed = crossfade_pcm(old, new)
        self.assertEqual(len(mixed), 800 * 4)


class SessionIntegrationTests(unittest.TestCase):
    class PlanEngine:
        def embed(self, _prompt):
            return np.zeros(768, dtype=np.float32)

    def test_new_take_resets_the_monitor(self):
        session = Session("floor-reset", "neutral", "guitar")
        votes = session.floor_monitor._drift_votes
        for _ in range(votes.maxlen):
            votes.append(True)
        self.assertTrue(session.floor_monitor.drifted)
        session.request_transport_reset()
        session.prepare_render()
        self.assertFalse(session.floor_monitor.drifted)

    def test_station_change_relearns_the_baseline(self):
        session = Session("floor-style", "neutral", "guitar")
        session.floor_monitor._baseline_floor = 0.123
        session.request_controls({"station": "rainy-piano"})
        session.conditioning_plan(self.PlanEngine(), 10)
        self.assertIsNone(session.floor_monitor._baseline_floor)

    def test_refresh_seeds_are_stable_and_distinct(self):
        session = Session("floor-seeds", "neutral", "guitar")
        first = session.next_refresh_seed()
        second = session.next_refresh_seed()
        self.assertNotEqual(first, second)
        self.assertEqual(session.refreshes, 2)
        again = Session("floor-seeds", "neutral", "guitar")
        self.assertEqual(again.next_refresh_seed(), first)

    def test_snapshot_reports_refreshes(self):
        session = Session("floor-snap", "neutral", "guitar")
        session.next_refresh_seed()
        self.assertEqual(session.snapshot()["takeRefreshes"], 1)


class WorkerRepairTests(unittest.TestCase):
    """The worker must splice onto a fresh state when a take drifts."""

    class RefreshEngine:
        size = "fake"
        backend = "fake"
        bits = 0
        temperature = 1.0
        top_k = 1
        cfg_musiccoca = 0.0
        cfg_notes = 0.0
        cfg_drums = 0.0
        style_token_levels = 0
        mlx_cache_mb = 0
        _fast = False
        _fast_sampling = False
        _fast_engine = None
        min_codebooks = 1
        max_codebooks = 1
        codebooks = 1
        target_rtf = 1.0
        load_error = None
        OLD = (np.full(4000, 8000, dtype=np.int16)).tobytes()
        FRESH = (np.full(4000, -8000, dtype=np.int16)).tobytes()

        def __init__(self):
            import threading

            self.is_ready = False
            self.gate = threading.Event()
            self.calls = []

        @property
        def ready(self):
            return self.is_ready

        def prepare_start(self):
            pass

        def request_stop(self):
            self.gate.set()

        def load(self):
            pass

        def warm_embeddings(self, _prompts, references=None):
            pass

        def calibrate(self):
            self.is_ready = True

        def embed(self, _prompt, _reference=None):
            return np.zeros(4, dtype=np.float32)

        def style_cache_key(self, prompt, reference=None):
            return prompt

        def generate(self, state, _plan, seed=None):
            self.gate.wait(5.0)
            self.calls.append((state, seed))
            if state is None and len(self.calls) > 1:
                return self.FRESH, "fresh-state"
            return self.OLD, "old-state"

        def note_render(self, *_args, **_kwargs):
            pass

        def realtime_factor(self):
            return 1.0

        def close(self):
            self.is_ready = False

    class StubMonitor:
        def __init__(self):
            self.resets = 0
            self._drifted = False
            self._armed = True

        def observe(self, _pcm):
            if self._armed:
                self._drifted = True

        @property
        def drifted(self):
            return self._drifted

        def describe(self):
            return "stub"

        def reset(self):
            self.resets += 1
            self._drifted = False
            self._armed = False

    def test_drifted_take_is_crossfaded_onto_a_fresh_state(self):
        import threading
        import session_manager as manager_mod

        manager = manager_mod.SessionManager()
        engine = self.RefreshEngine()
        manager.engine = engine
        delivered = []
        got_pcm = threading.Event()

        def epoch_sink(pcm, _epoch):
            delivered.append(pcm)
            got_pcm.set()

        manager.start()
        try:
            deadline = time.monotonic() + 5.0
            while not engine.ready and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(engine.ready)

            session, _ = manager.attach(
                None, "neutral", "guitar", epoch_sink=epoch_sink
            )
            stub = self.StubMonitor()
            session.floor_monitor = stub
            original_seed = session.seed
            engine.gate.set()

            self.assertTrue(got_pcm.wait(5.0))
            deadline = time.monotonic() + 5.0
            while stub.resets == 0 and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertEqual(stub.resets, 1)
            self.assertEqual(
                delivered[0], crossfade_pcm(engine.OLD, engine.FRESH)
            )
            # the fresh render started from no state, under a refresh seed
            fresh_calls = [c for c in engine.calls if c[0] is None and c[1] is not None]
            self.assertGreaterEqual(len(fresh_calls), 2)  # first take + refresh
            self.assertEqual(session.seed, fresh_calls[1][1])
            self.assertNotEqual(session.seed, original_seed)
            self.assertEqual(session.refreshes, 1)
        finally:
            manager.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
