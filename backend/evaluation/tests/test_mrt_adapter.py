import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.mrt_adapter import MRTEngineAdapter  # noqa: E402
from evaluation.spec import evaluation_spec_from_dict  # noqa: E402


class FakeMRTEngine:
    instances = []

    def __init__(self):
        self._fast = False
        self._fast_sampling = True
        self.codebooks = 12
        self.max_codebooks = 12
        self.closed = False
        self.generated = []
        FakeMRTEngine.instances.append(self)

    def prepare_start(self):
        pass

    def load(self):
        self._fast = True

    def warm_embeddings(self, prompts):
        self.warmed = list(prompts)

    def embed(self, _prompt, _reference=None):
        return np.zeros(768, dtype=np.float32)

    @staticmethod
    def style_cache_key(prompt, reference=None):
        return f"{prompt}:{reference}"

    def set_codebooks(self, value):
        self.codebooks = value

    def generate(self, _state, plan, seed=None):
        frames = sum(run.frames for run in plan)
        self.generated.append((plan, seed))
        return bytes(frames * 1920 * 2 * 2), object()

    def close(self):
        self.closed = True


class FakeGuide:
    def __init__(self, key):
        self.key = key
        self.seed = 777
        self.tonic = 60

    def configure(self, **values):
        self.controls = values

    def event_frames(self, mood, frames):
        tokens = list((-1,) * 128)
        tokens[60] = 2
        onset = SimpleNamespace(tokens=tuple(tokens), drum=1, mood=mood)
        tokens[60] = 1
        sustain = SimpleNamespace(tokens=tuple(tokens), drum=-1, mood=mood)
        return [onset] + [sustain] * (frames - 1)


class FakeSamplingControls:
    def __init__(self, **values):
        self.__dict__.update(values)


class FakeConditioningRun:
    def __init__(self, **values):
        self.__dict__.update(values)


def fake_modules():
    engine = ModuleType("engine")
    engine.ABSOLUTE_MIN_CODEBOOKS = 8
    engine.MRTEngine = FakeMRTEngine
    engine.SamplingControls = FakeSamplingControls
    engine.ConditioningRun = FakeConditioningRun
    melody = ModuleType("melody")
    melody.MelodyGuide = FakeGuide
    return {"engine": engine, "melody": melody}


class MRTEngineAdapterTests(unittest.TestCase):
    def test_applies_explicit_config_and_uses_production_event_plan(self):
        FakeMRTEngine.instances.clear()
        spec = evaluation_spec_from_dict(
            {
                "duration_seconds": 0.4,
                "warmup_seconds": 0.08,
                "engine": {"model_size": "mrt2_base", "bits": 4},
                "seeds": [123],
                "styles": [
                    {
                        "id": "style",
                        "prompt": "soft instrumental",
                        "mood": "somber",
                        "intensity": 0.8,
                    }
                ],
                "presets": [
                    {
                        "id": "preset",
                        "temperature": 1.1,
                        "top_k": 50,
                        "cfg_musiccoca": 1.6,
                        "cfg_notes": 2.4,
                        "cfg_drums": 4.0,
                        "sampling_mode": "production",
                    }
                ],
                "composition_seed": 999,
                "note_modes": [
                    "app_melody",
                    "masked",
                    {"id": "drone", "kind": "constant", "midi_note": 64},
                ],
                "codebook_depths": [12],
            }
        )
        adapter = MRTEngineAdapter(spec.engine, spec.presets[0])
        with patch.dict(sys.modules, fake_modules()):
            adapter.prepare(spec.styles)
            cases = {case.note_mode.kind: case for case in spec.expand()}
            result = adapter.render(cases["app_melody"])
            engine = FakeMRTEngine.instances[-1]

            self.assertEqual(engine.size, "mrt2_base")
            self.assertEqual(engine.bits, 4)
            self.assertEqual(engine.temperature, 1.1)
            self.assertEqual(engine.cfg_notes, 2.4)
            self.assertEqual(engine.style_token_levels, 6)
            self.assertEqual(engine.warmed, ["soft instrumental"])
            plan, seed = engine.generated[0]
            self.assertEqual(seed, 123)
            self.assertIsInstance(plan[0], FakeConditioningRun)
            self.assertEqual(plan[0].drum, 1)
            self.assertEqual(plan[0].notes[60], 2)
            self.assertEqual(plan[1].notes[60], 1)
            self.assertAlmostEqual(plan[0].sampling.temperature, 1.1)
            self.assertEqual(plan[0].sampling.top_k, 50)
            self.assertAlmostEqual(plan[0].sampling.cfg_notes, 2.4)
            self.assertEqual(len(result.pcm_s16le), 12 * 1920 * 2 * 2)

            masked, _ = adapter._plan(cases["masked"])
            guided_drums = [run.drum for run in plan for _ in range(run.frames)]
            masked_drums = [run.drum for run in masked for _ in range(run.frames)]
            self.assertEqual(masked_drums, guided_drums)
            self.assertTrue(all(token == -1 for run in masked for token in run.notes))

            constant, _ = adapter._plan(cases["constant"])
            constant_notes = [run.notes for run in constant for _ in range(run.frames)]
            self.assertEqual(constant_notes[0][64], 2)
            self.assertTrue(all(notes[64] == 1 for notes in constant_notes[1:]))
            adapter.close()
            self.assertTrue(engine.closed)

    def test_builds_current_conditioning_runs_without_loading_a_model(self):
        spec = evaluation_spec_from_dict(
            {
                "duration_seconds": 0.4,
                "warmup_seconds": 0.0,
                "engine": {"model_size": "mrt2_small"},
                "seeds": [55],
                "composition_seed": 77,
                "styles": [{"id": "station", "station": "dusty-beats"}],
                "presets": [
                    {
                        "id": "production",
                        "temperature": 1.1,
                        "top_k": 50,
                        "cfg_musiccoca": 1.6,
                        "cfg_notes": 2.4,
                        "cfg_drums": 4.0,
                        "sampling_mode": "production",
                    }
                ],
                "note_modes": ["app_melody"],
                "drum_modes": ["planned"],
                "codebook_depths": [12],
            }
        )

        class PlanningEngine:
            @staticmethod
            def embed(_prompt, _reference=None):
                return np.zeros(768, dtype=np.float32)

            @staticmethod
            def style_cache_key(prompt, reference=None):
                return f"{prompt}:{reference}"

        engine_module = importlib.import_module("engine")
        melody_module = importlib.import_module("melody")
        adapter = MRTEngineAdapter(spec.engine, spec.presets[0])
        adapter._engine = PlanningEngine()
        adapter._engine_module = engine_module
        adapter._melody_module = melody_module

        plan, metadata = adapter._plan(spec.expand()[0])

        self.assertTrue(all(isinstance(run, engine_module.ConditioningRun) for run in plan))
        self.assertEqual(sum(run.frames for run in plan), 10)
        self.assertEqual(metadata["composition_seed"], 77)
        self.assertEqual(metadata["station"], "dusty-beats")
        self.assertTrue(any(2 in run.notes for run in plan))


if __name__ == "__main__":
    unittest.main()
