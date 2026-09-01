"""Conditioning invariants that do not require loading MRT2 or MLX."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import MRTEngine  # noqa: E402
from session import Session  # noqa: E402
import styles  # noqa: E402


class FakeStyleModel:
    def tokenize(self, _style):
        return np.arange(12, dtype=np.int32)


class FakeSystem:
    def __init__(self):
        self._style_model = FakeStyleModel()
        self.calls = []

    def _build_conditioning(self, conditioning, *_args):
        copied = {key: list(value) for key, value in conditioning.items()}
        self.calls.append(copied)
        return copied, {"constants": True}


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        self.engine = MRTEngine()
        self.engine._style_key = "style"
        self.engine._notes_key = "notes"
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

    def test_session_combines_style_and_note_boundaries_without_losing_frames(self):
        class PlanEngine:
            def embed(self, _prompt):
                return np.zeros(768, dtype=np.float32)

        session = Session("conditioning-plan", "neutral", "guitar")
        plan = session.conditioning_plan(PlanEngine(), 75)

        self.assertEqual(sum(run[3] for run in plan), 75)
        self.assertTrue(any(note is not None for _style, _key, note, _n in plan))
        self.assertTrue(all(frames > 0 for _style, _key, _note, frames in plan))
        self.assertEqual(session.conditioning_plan(PlanEngine(), 0), [])

    def test_every_style_prompt_explicitly_targets_melody_and_clean_mix(self):
        for prompt in styles.all_prompts():
            self.assertIn("melodic instrumental", prompt)
            self.assertIn("lead", prompt)
            self.assertIn("coherent jazz harmony", prompt)
            self.assertIn("clean balanced mix", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
