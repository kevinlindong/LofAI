import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.report import (  # noqa: E402
    decode_pairwise_results,
    generate_pairwise_report,
)
from evaluation.runner import (  # noqa: E402
    AdapterRender,
    read_pcm16_wav,
    run_evaluation,
)
from evaluation.shortlist import (  # noqa: E402
    SHORTLIST_DISCLAIMER,
    build_signal_shortlist,
)
from evaluation.spec import evaluation_spec_from_dict  # noqa: E402


SAMPLE_RATE = 48_000
SAMPLES_PER_FRAME = 1_920


def tiny_spec():
    return evaluation_spec_from_dict(
        {
            "duration_seconds": 0.4,
            "warmup_seconds": 0.08,
            "engine": {"model_size": "mrt2_small"},
            "seeds": [17, 18],
            "styles": [
                {"id": "one-style", "prompt": "gentle instrumental", "mood": "neutral"}
            ],
            "presets": [
                {
                    "id": "a",
                    "temperature": 1.0,
                    "top_k": 40,
                    "cfg_musiccoca": 1.5,
                    "cfg_notes": 1.0,
                    "cfg_drums": 1.0,
                },
                {
                    "id": "b",
                    "temperature": 1.1,
                    "top_k": 50,
                    "cfg_musiccoca": 2.0,
                    "cfg_notes": 2.0,
                    "cfg_drums": 1.0,
                },
            ],
            "note_modes": ["masked"],
            "codebook_depths": [12],
        }
    )


class FakeAdapter:
    def __init__(self, preset, calls):
        self.preset = preset
        self.calls = calls
        self.closed = False

    def prepare(self, styles):
        self.calls.append(("prepare", self.preset.id, tuple(style.id for style in styles)))

    def render(self, case):
        self.calls.append(("render", case.case_id))
        sample_count = case.total_frames * SAMPLES_PER_FRAME
        time = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
        frequency = 220 + (case.seed % 5) * 30 + (20 if self.preset.id == "b" else 0)
        mono = 0.15 * np.sin(2 * np.pi * frequency * time)
        pcm = np.ascontiguousarray(
            np.column_stack((mono, mono)) * 32767.0, dtype=np.int16
        ).tobytes()
        return AdapterRender(pcm, metadata={"mock": True, "seed": case.seed})

    def close(self):
        self.closed = True
        self.calls.append(("close", self.preset.id))


class RunnerAndReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "run"
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def factory(self, _engine, preset):
        return FakeAdapter(preset, self.calls)

    def test_mock_render_writes_wav_json_csv_and_resume_skips(self):
        spec = tiny_spec()
        manifest = run_evaluation(spec, self.output, self.factory, progress=None)

        self.assertEqual(len(manifest["cases"]), 4)
        self.assertTrue(all(case["status"] == "complete" for case in manifest["cases"]))
        self.assertTrue((self.output / "manifest.json").is_file())
        self.assertTrue((self.output / "manifest.csv").is_file())
        for case in manifest["cases"]:
            wav = self.output / case["wav_path"]
            samples, rate = read_pcm16_wav(wav)
            self.assertEqual(rate, SAMPLE_RATE)
            self.assertEqual(samples.shape, (10 * SAMPLES_PER_FRAME, 2))
            self.assertIn("spectral_flatness", case["metrics"])

        render_calls = len([call for call in self.calls if call[0] == "render"])
        run_evaluation(spec, self.output, self.factory, resume=True, progress=None)
        self.assertEqual(
            len([call for call in self.calls if call[0] == "render"]), render_calls
        )

    def test_resume_rerenders_hash_mismatch_only(self):
        spec = tiny_spec()
        manifest = run_evaluation(spec, self.output, self.factory, progress=None)
        broken = self.output / manifest["cases"][0]["wav_path"]
        broken.write_bytes(b"not a wav")
        before = len([call for call in self.calls if call[0] == "render"])

        run_evaluation(spec, self.output, self.factory, resume=True, progress=None)

        after = len([call for call in self.calls if call[0] == "render"])
        self.assertEqual(after, before + 1)

    def test_blind_report_hides_variants_and_decodes_responses(self):
        run_evaluation(tiny_spec(), self.output, self.factory, progress=None)
        report_path, key_path = generate_pairwise_report(
            self.output / "manifest.json", seed=99, max_pairs=20
        )
        html = report_path.read_text(encoding="utf-8")
        key = json.loads(key_path.read_text(encoding="utf-8"))

        self.assertNotIn("quality-first", html)
        self.assertNotIn("__a__", html)
        self.assertNotIn("__b__", html)
        self.assertGreaterEqual(len(key["trials"]), 2)
        first = key["trials"][0]
        responses = {
            "listener_id": "listener-x",
            "answers": [
                {"trial_id": first["trial_id"], "choice": "b", "note": "more coherent"}
            ],
        }
        response_path = self.output / "response.json"
        response_path.write_text(json.dumps(responses), encoding="utf-8")

        decoded = decode_pairwise_results(key_path, response_path)

        self.assertEqual(decoded[0]["winner_case_id"], first["b_case_id"])
        self.assertEqual(decoded[0]["listener_id"], "listener-x")

        take_report, take_key = generate_pairwise_report(
            self.output / "manifest.json",
            seed=100,
            max_pairs=20,
            hold_seed_constant=False,
        )
        self.assertTrue(take_report.is_file())
        self.assertFalse(
            json.loads(take_key.read_text(encoding="utf-8"))["hold_seed_constant"]
        )

    def test_shortlist_is_explicitly_non_authoritative(self):
        run_evaluation(tiny_spec(), self.output, self.factory, progress=None)
        result = build_signal_shortlist(
            self.output / "manifest.json", count=2
        )

        self.assertEqual(len(result["shortlist"]), 2)
        self.assertIn("Non-authoritative", SHORTLIST_DISCLAIMER)
        self.assertIn("cannot assess melody", result["disclaimer"])
        self.assertTrue((self.output / "signal_shortlist.json").is_file())


if __name__ == "__main__":
    unittest.main()
