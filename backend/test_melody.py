"""Compatibility and adapter checks for the MRT2 melody surface."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import melody  # noqa: E402
from composition import PianoRollEvent  # noqa: E402


def expand(runs):
    values = []
    for value, frames in runs:
        values.extend([value] * frames)
    return values


class MelodyGuideCompatibilityTests(unittest.TestCase):
    def test_legacy_plan_is_deterministic_and_chunk_independent(self):
        whole = expand(melody.MelodyGuide("same-session").plan("neutral", 500))
        split_guide = melody.MelodyGuide("same-session")
        split = expand(split_guide.plan("neutral", 137))
        split += expand(split_guide.plan("neutral", 211))
        split += expand(split_guide.plan("neutral", 152))

        self.assertEqual(split, whole)
        self.assertTrue(any(note is not None for note in whole))

    def test_new_event_plan_preserves_exact_mrt_tokens(self):
        guide = melody.MelodyGuide("exact-events")
        events = expand(guide.plan_events("neutral", 80))

        self.assertEqual(len(events), 80)
        self.assertTrue(any(2 in event.tokens for event in events))
        self.assertTrue(any(1 in event.tokens for event in events))
        self.assertTrue(any(0 in event.tokens for event in events))
        self.assertTrue(all(3 not in event.tokens for event in events))

    def test_integer_piano_roll_keeps_legacy_autostrum_behavior(self):
        self.assertIsNone(melody.piano_roll(None))
        tokens = melody.piano_roll(64)

        self.assertEqual(len(tokens), 128)
        self.assertEqual(tokens[64], 3)
        self.assertEqual(tokens.count(3), 1)
        self.assertEqual(tokens.count(-1), 127)
        with self.assertRaises(ValueError):
            melody.piano_roll(128)

    def test_event_and_raw_roll_adapters_validate_without_losing_states(self):
        tokens = [-1] * 128
        tokens[60] = 0
        tokens[62] = 1
        tokens[64] = 2
        event = PianoRollEvent(tuple(tokens), drum=1)

        self.assertEqual(melody.piano_roll(event), tokens)
        self.assertEqual(melody.piano_roll(tuple(tokens)), tokens)
        self.assertEqual(melody.drum_intent(event), 1)
        self.assertEqual(melody.drum_intent(0), 0)
        with self.assertRaises(ValueError):
            melody.piano_roll(tokens[:-1])
        with self.assertRaises(ValueError):
            melody.piano_roll([4] * 128)
        with self.assertRaises(ValueError):
            melody.drum_intent(2)

    def test_runtime_configuration_does_not_reset_the_legacy_clock(self):
        guide = melody.MelodyGuide("runtime-controls")
        guide.plan("neutral", 31)
        before = guide.frame_index
        guide.configure(
            station="rainy cafe",
            bpm=86,
            groove=0.8,
            intensity=0.7,
            melody_enabled=True,
            drums_enabled=False,
        )
        guide.plan("neutral", 17)

        self.assertEqual(before, 31)
        self.assertEqual(guide.frame_index, 48)
        self.assertEqual(guide.composition.settings.station, "rainy cafe")
        self.assertEqual(guide.composition.settings.bpm, 86)

    def test_wire_toggle_names_are_accepted_by_the_facade(self):
        guide = melody.MelodyGuide("wire-controls")
        settings = guide.configure(
            station="quiet-library",
            mood="somber",
            instrument="rhodes",
            bpm=72,
            groove=0.7,
            intensity=0.3,
            melody=False,
            drums=False,
        )

        self.assertFalse(settings.melody_enabled)
        self.assertFalse(settings.drums_enabled)
        self.assertEqual(settings.station, "quiet-library")
        self.assertEqual(settings.mood, "somber")

    def test_session_seed_and_key_are_repeatable_but_not_globally_fixed(self):
        first = melody.MelodyGuide("listener-a")
        again = melody.MelodyGuide("listener-a")
        others = [melody.MelodyGuide(f"listener-{index}") for index in range(12)]

        self.assertEqual(first.seed, again.seed)
        self.assertEqual(first.tonic, again.tonic)
        self.assertGreater(len({guide.tonic for guide in others}), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
