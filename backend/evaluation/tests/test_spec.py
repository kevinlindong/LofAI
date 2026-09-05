import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.spec import evaluation_spec_from_dict  # noqa: E402


def valid_config():
    return {
        "schema_version": 1,
        "duration_seconds": 0.4,
        "warmup_seconds": 0.08,
        "engine": {"model_size": "mrt2_small", "backend": "python", "bits": 8},
        "seeds": [7, 11],
        "styles": [
            {"id": "warm", "prompt": "warm instrumental jazz", "mood": "neutral"}
        ],
        "presets": [
            {
                "id": "focused",
                "temperature": 1.1,
                "top_k": 50,
                "cfg_musiccoca": 1.6,
                "cfg_notes": 2.4,
                "cfg_drums": 4.0,
            },
            {
                "id": "wide",
                "temperature": 1.2,
                "top_k": 100,
                "cfg_musiccoca": 2.0,
                "cfg_notes": 1.0,
                "cfg_drums": 1.0,
                "style_token_levels": 12,
            },
        ],
        "note_modes": ["masked", {"id": "guide", "kind": "app_melody"}],
        "codebook_depths": [12, 10],
    }


class EvaluationSpecTests(unittest.TestCase):
    def test_expands_full_matrix_with_stable_ids(self):
        first = evaluation_spec_from_dict(valid_config())
        second = evaluation_spec_from_dict(valid_config())

        self.assertEqual(len(first.expand()), 16)
        self.assertEqual(
            [case.case_id for case in first.expand()],
            [case.case_id for case in second.expand()],
        )
        self.assertEqual(first.duration_frames, 10)
        self.assertEqual(first.warmup_frames_for(first.styles[0]), 2)
        self.assertTrue(all(case.total_frames == 12 for case in first.expand()))

    def test_case_id_changes_when_audio_setting_changes(self):
        original = evaluation_spec_from_dict(valid_config()).expand()[0].case_id
        changed = valid_config()
        changed["presets"][0]["temperature"] = 1.11
        updated = evaluation_spec_from_dict(changed).expand()[0].case_id

        self.assertNotEqual(original, updated)

    def test_rejects_non_frame_aligned_duration_and_unknown_fields(self):
        invalid_duration = valid_config()
        invalid_duration["duration_seconds"] = 0.41
        with self.assertRaisesRegex(ValueError, "multiple of 0.04"):
            evaluation_spec_from_dict(invalid_duration)

        unknown = valid_config()
        unknown["mystery"] = True
        with self.assertRaisesRegex(ValueError, "unknown top-level"):
            evaluation_spec_from_dict(unknown)

        invalid_bits = valid_config()
        invalid_bits["engine"]["bits"] = 16
        with self.assertRaisesRegex(ValueError, "full precision"):
            evaluation_spec_from_dict(invalid_bits)

    def test_validates_constant_note_and_unique_dimensions(self):
        constant = valid_config()
        constant["note_modes"] = [
            {"id": "drone", "kind": "constant", "midi_note": 60}
        ]
        spec = evaluation_spec_from_dict(constant)
        self.assertEqual(spec.note_modes[0].midi_note, 60)

        duplicate = valid_config()
        duplicate["seeds"] = [7, 7]
        with self.assertRaisesRegex(ValueError, "seeds must be unique"):
            evaluation_spec_from_dict(duplicate)

    def test_named_station_resolves_live_controls_and_bar_aligned_warmup(self):
        config = valid_config()
        config.pop("warmup_seconds")
        config["warmup_bars"] = 1
        config["styles"] = [{"id": "station", "station": "dusty-beats"}]

        spec = evaluation_spec_from_dict(config)
        style = spec.styles[0]
        case = spec.expand()[0]

        self.assertEqual(
            style.prompt,
            "instrumental mellow lo-fi hip hop, dusty drums, warm jazz guitar, vinyl",
        )
        self.assertEqual(style.bpm, 76)
        self.assertEqual(style.groove, 0.62)
        self.assertEqual(case.warmup_bars, 1)
        self.assertLess(case.listening_start_step_in_bar, 0.05)

    def test_reference_hash_and_drum_modes_are_case_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "owned.wav"
            reference.write_bytes(b"owned-reference-fixture")
            config = valid_config()
            config["styles"] = [
                {
                    "id": "anchored",
                    "station": "dusty-beats",
                    "audio_reference": str(reference),
                }
            ]
            config["drum_modes"] = ["planned", "strict", "masked", "off"]
            spec = evaluation_spec_from_dict(config)

            self.assertEqual(
                spec.styles[0].audio_reference_sha256,
                hashlib.sha256(reference.read_bytes()).hexdigest(),
            )
            self.assertEqual(len(spec.expand()), 64)
            self.assertEqual(
                {case.drum_mode.kind for case in spec.expand()},
                {"planned", "strict", "masked", "off"},
            )


if __name__ == "__main__":
    unittest.main()
