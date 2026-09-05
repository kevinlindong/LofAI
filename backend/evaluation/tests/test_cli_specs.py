from argparse import Namespace
from pathlib import Path
import sys
import unittest

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.cli import _quality_spec  # noqa: E402


def args(**overrides):
    values = {
        "model_size": "mrt2_base",
        "bits": 8,
        "duration_seconds": 60.0,
        "warmup_seconds": 2.0,
        "warmup_bars": None,
        "station": "custom",
        "prompt": "coherent instrumental music",
        "mood": "neutral",
        "instrument": "guitar",
        "bpm": 78,
        "groove": 0.6,
        "intensity": 0.5,
        "melody": True,
        "drums": True,
        "note_mode": "app_melody",
        "drum_mode": "planned",
        "audio_reference": None,
        "text_only": True,
        "audio_style_blend": 0.75,
        "temperature": 1.1,
        "top_k": 50,
        "cfg_musiccoca": 1.6,
        "cfg_notes": 2.4,
        "cfg_drums": 4.0,
        "style_token_levels": 6,
        "sampling_mode": "production",
        "composition_seed": None,
        "seed": 42,
        "takes": 4,
        "seed_start": 100,
        "report_seed": 99,
        "max_pairs": 20,
        "shortlist_count": 3,
    }
    values.update(overrides)
    return Namespace(**values)


class ConvenienceSpecTests(unittest.TestCase):
    def test_single_candidate_is_base_full_depth_and_fixed_seed(self):
        spec = _quality_spec(args(), multi=False)

        self.assertEqual(spec.engine.model_size, "mrt2_base")
        self.assertEqual(spec.codebook_depths, (12,))
        self.assertEqual(spec.seeds, (42,))
        self.assertEqual(spec.duration_seconds, 60.0)
        self.assertEqual(spec.drum_modes[0].kind, "planned")

    def test_multi_take_changes_only_seed(self):
        spec = _quality_spec(args(), multi=True)
        cases = spec.expand()

        self.assertEqual(spec.seeds, (100, 101, 102, 103))
        self.assertEqual(len(cases), 4)
        self.assertTrue(all(case.codebook_depth == 12 for case in cases))

        fixed = _quality_spec(args(composition_seed=999), multi=True)
        self.assertTrue(all(case.composition_seed == 999 for case in fixed.expand()))

    def test_convenience_duration_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "between 30 and 90"):
            _quality_spec(args(duration_seconds=29.0), multi=False)
        with self.assertRaisesRegex(ValueError, "between 30 and 90"):
            _quality_spec(args(duration_seconds=91.0), multi=True)


if __name__ == "__main__":
    unittest.main()
