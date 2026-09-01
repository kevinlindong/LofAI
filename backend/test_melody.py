"""Deterministic musical-structure checks for the MRT2 melody guide."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import melody  # noqa: E402


def expand(runs):
    values = []
    for value, frames in runs:
        values.extend([value] * frames)
    return values


class MelodyGuideTests(unittest.TestCase):
    def test_chunking_does_not_change_the_phrase(self):
        whole = expand(melody.MelodyGuide("same-session").plan("neutral", 500))
        split_guide = melody.MelodyGuide("same-session")
        split = expand(split_guide.plan("neutral", 137))
        split += expand(split_guide.plan("neutral", 211))
        split += expand(split_guide.plan("neutral", 152))

        self.assertEqual(split, whole)

    def test_every_mood_is_tonal_restrained_and_resolving(self):
        for mood, scale in melody.SCALES.items():
            guide = melody.MelodyGuide(f"quality-{mood}")
            notes = expand(guide.plan(mood, melody.FRAME_RATE * 60))
            sounding = [note for note in notes if note is not None]
            pitches = sorted(set(sounding))

            self.assertGreater(len(sounding), len(notes) * 0.25, mood)
            self.assertLess(len(sounding), len(notes) * 0.80, mood)
            self.assertGreaterEqual(len(pitches), 5, mood)
            self.assertTrue(
                all((note - guide.tonic) in scale for note in pitches),
                (mood, pitches, guide.tonic),
            )
            transitions = [
                (a, b)
                for a, b in zip(notes, notes[1:])
                if a is not None and b is not None and a != b
            ]
            self.assertTrue(
                all(abs(a - b) <= 12 for a, b in transitions), transitions
            )
            for phrase_index in range(4):
                phrase = guide._phrase(mood, phrase_index)
                self.assertEqual(phrase[-2:], (0, None), (mood, phrase_index))

    def test_seed_is_repeatable_but_sessions_are_not_all_identical(self):
        a = melody.MelodyGuide("listener-a")
        again = melody.MelodyGuide("listener-a")
        others = [melody.MelodyGuide(f"listener-{i}") for i in range(8)]

        self.assertEqual(a.seed, again.seed)
        self.assertEqual(a.tonic, again.tonic)
        self.assertGreater(len({guide.tonic for guide in others}), 1)

    def test_piano_roll_masks_accompaniment_and_marks_one_guide_note(self):
        self.assertIsNone(melody.piano_roll(None))
        tokens = melody.piano_roll(64)

        self.assertEqual(len(tokens), 128)
        self.assertEqual(tokens[64], 3)
        self.assertEqual(tokens.count(3), 1)
        self.assertEqual(tokens.count(-1), 127)
        with self.assertRaises(ValueError):
            melody.piano_roll(128)

    def test_mood_change_waits_for_a_step_boundary(self):
        guide = melody.MelodyGuide("mood-change")
        guide.plan("somber", 3)
        old_mood = guide._active_mood
        guide.plan("lively", 1)
        self.assertEqual(guide._active_mood, old_mood)

        for _ in range(20):
            guide.plan("lively", 1)
            if guide._active_mood == "lively":
                break
        self.assertEqual(guide._active_mood, "lively")


if __name__ == "__main__":
    unittest.main(verbosity=2)
