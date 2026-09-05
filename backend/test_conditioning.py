"""Conditioning invariants that do not require loading MRT2 or MLX."""

import os
import sys
import unittest
from dataclasses import astuple
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import ConditioningRun, MRTEngine, SamplingControls  # noqa: E402
from composition import PianoRollEvent  # noqa: E402
from session import Session  # noqa: E402
import styles  # noqa: E402


class FakeStyleModel:
    def tokenize(self, _style):
        return np.arange(12, dtype=np.int32)


class FakeSystem:
    def __init__(self):
        self._style_model = FakeStyleModel()
        self.calls = []
        self.parameters = []

    def _build_conditioning(self, conditioning, *args):
        copied = {key: list(value) for key, value in conditioning.items()}
        self.calls.append(copied)
        self.parameters.append(args)
        return copied, {"constants": True}


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        self.engine = MRTEngine()
        self.engine._style_key = "style"
        self.engine._notes_key = "notes"
        self.engine._drums_key = "drums"
        self.engine._system = FakeSystem()
        self.engine.style_token_levels = 6
        self.style = np.zeros(768, dtype=np.float32)

    def test_masks_fine_style_tokens_and_encodes_the_guide_note(self):
        block, _ = self.engine._conditioning(self.style, "prompt", 64)

        self.assertEqual(block["style"][:6], list(range(6)))
        self.assertEqual(block["style"][6:], [-1] * 6)
        self.assertEqual(block["notes"].count(3), 1)
        self.assertEqual(block["notes"][64], 3)
        self.assertEqual(block["notes"].count(-1), 127)

    def test_cache_key_includes_note_so_stale_pitch_cannot_be_reused(self):
        first = self.engine._conditioning(self.style, "prompt", 60)
        same = self.engine._conditioning(self.style, "prompt", 60)
        changed = self.engine._conditioning(self.style, "prompt", 62)

        self.assertIs(first, same)
        self.assertIsNot(first, changed)
        self.assertEqual(len(self.engine._system.calls), 2)

    def test_a_rest_leaves_the_entire_note_input_masked(self):
        block, _ = self.engine._conditioning(self.style, "prompt", None)

        self.assertNotIn("notes", block)

    def test_exact_score_drums_and_per_listener_sampling_reach_mrt(self):
        tokens = [-1] * 128
        tokens[60], tokens[64], tokens[67] = 2, 1, 0
        event = PianoRollEvent(tuple(tokens), drum=1)
        sampling = SamplingControls(1.15, 48, 1.7, 2.2, 3.5)

        block, _ = self.engine._conditioning(
            self.style, "exact", event, sampling=sampling
        )

        self.assertEqual(block["notes"], tokens)
        self.assertEqual(block["drums"], [1])
        cfg, temperature, top_k = self.engine._system.parameters[-1]
        self.assertEqual(cfg, {"musiccoca": 1.7, "notes": 2.2, "drums": 3.5})
        self.assertEqual(temperature, 1.15)
        self.assertEqual(top_k, 48)

    def test_tuple_compatibility_uses_conditioning_run_field_order(self):
        sampling = SamplingControls(1.1, 50, 1.6, 2.4, 4.0)
        run = ConditioningRun(self.style, "tuple", None, 25, 1, sampling)

        _style, _key, _notes, frames, drum, parsed = self.engine._run_parts(
            astuple(run)
        )

        self.assertEqual(frames, 25)
        self.assertEqual(drum, 1)
        self.assertEqual(parsed, sampling)

    def test_stock_eager_fallback_receives_the_requested_decoder_seed(self):
        class FakeStockSystem:
            _style_model = FakeStyleModel()

            def __init__(self):
                self.states = []

            def generate(self, *, frames, state, **_kwargs):
                self.states.append(state)
                samples = np.zeros((frames * 1920, 2), dtype=np.float32)
                return SimpleNamespace(samples=samples), "next-state"

        system = FakeStockSystem()
        self.engine._system = system
        self.engine._fast = False
        run = ConditioningRun(self.style, "seeded", None, 2)
        seeded_state = object()

        with patch.object(self.engine, "_new_eager_state", return_value=seeded_state) as seed:
            _pcm, state = self.engine.generate(None, (run,), seed=9876)

        seed.assert_called_once_with(9876)
        self.assertIs(system.states[0], seeded_state)
        self.assertEqual(state, "next-state")

    def test_invalid_quantization_width_fails_before_model_construction(self):
        self.engine.bits = 16
        with self.assertRaisesRegex(ValueError, "full precision"):
            self.engine._load_python()

    def test_session_combines_style_score_and_drum_boundaries_without_losing_frames(self):
        class PlanEngine:
            def embed(self, _prompt):
                return np.zeros(768, dtype=np.float32)

        session = Session("conditioning-plan", "neutral", "guitar")
        plan = session.conditioning_plan(PlanEngine(), 75)

        self.assertEqual(sum(run.frames for run in plan), 75)
        self.assertTrue(any(2 in run.notes.tokens for run in plan))
        self.assertTrue(any(run.drum in (-1, 1) for run in plan))
        self.assertTrue(all(run.frames > 0 for run in plan))
        self.assertEqual(session.conditioning_plan(PlanEngine(), 0), [])

    def test_style_prompts_are_short_audible_tags_not_mix_instructions(self):
        for prompt in styles.all_prompts():
            self.assertLessEqual(len(prompt.split()), 12)
            self.assertNotIn("coherent", prompt)
            self.assertNotIn("balanced mix", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
