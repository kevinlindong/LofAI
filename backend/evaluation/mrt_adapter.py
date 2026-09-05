"""Adapter from the evaluation runner to lofAI's production ``MRTEngine``."""

from __future__ import annotations

import importlib
import hashlib
from pathlib import Path
import sys
from typing import Sequence

from .runner import AdapterRender
from .spec import EngineSpec, ParameterPreset, RenderCase, StyleSpec


class MRTEngineAdapter:
    """Render cases through the same engine method used by live sessions.

    Runtime imports are intentionally delayed until ``prepare`` so importing
    evaluation modules and running mock-backed tests never imports MLX or
    Magenta RealTime.
    """

    def __init__(self, engine_spec: EngineSpec, preset: ParameterPreset):
        self.engine_spec = engine_spec
        self.preset = preset
        self._engine = None
        self._engine_module = None
        self._melody_module = None

    def prepare(self, styles: Sequence[StyleSpec]) -> None:
        backend_dir = Path(__file__).resolve().parents[1]
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        engine_module = importlib.import_module("engine")
        self._engine_module = engine_module
        self._melody_module = importlib.import_module("melody")

        engine = engine_module.MRTEngine()
        # Assign every evaluated knob explicitly. Ambient MRT_* environment
        # variables must not silently change a supposedly reproducible run.
        engine.size = self.engine_spec.model_size
        engine.backend = self.engine_spec.backend
        engine.bits = self.engine_spec.bits
        engine.fast_sampler_enabled = self.engine_spec.fast_sampler
        engine.audio_style_blend = self.engine_spec.audio_style_blend
        engine.temperature = self.preset.temperature
        engine.top_k = self.preset.top_k
        engine.cfg_musiccoca = self.preset.cfg_musiccoca
        engine.cfg_notes = self.preset.cfg_notes
        engine.cfg_drums = self.preset.cfg_drums
        engine.style_token_levels = self.preset.style_token_levels
        engine.pinned_codebooks = 12
        engine.min_codebooks = engine_module.ABSOLUTE_MIN_CODEBOOKS
        self._engine = engine
        try:
            engine.prepare_start()
            engine.load()
            if not engine._fast:
                raise RuntimeError(
                    "evaluation requires MRTEngine's eager fast path to pin and "
                    "record the requested decoder codebook depth"
                )
            prompts = list(dict.fromkeys(style.prompt for style in styles))
            engine.warm_embeddings(prompts)
            for style in styles:
                if style.audio_reference is not None:
                    self._verify_reference(style)
                    engine.embed(style.prompt, style.audio_reference)
        except Exception:
            engine.close()
            self._engine = None
            raise

    @staticmethod
    def _verify_reference(style: StyleSpec) -> None:
        if style.audio_reference is None:
            return
        digest = hashlib.sha256()
        with Path(style.audio_reference).open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != style.audio_reference_sha256:
            raise RuntimeError(
                f"audio reference changed after planning evaluation: {style.audio_reference}"
            )

    def _sampling(self, case: RenderCase):
        if self._engine_module is None:
            raise RuntimeError("adapter has not been prepared")
        preset = case.preset
        if preset.sampling_mode == "production":
            # Production keeps stochasticity stable and expresses intensity in
            # the composition/arrangement. Keep this explicit branch so future
            # candidate transforms remain visible in evaluation manifests.
            temperature = preset.temperature
            top_k = preset.top_k
            cfg_musiccoca = preset.cfg_musiccoca
            cfg_notes = preset.cfg_notes
            cfg_drums = preset.cfg_drums
        else:
            temperature = preset.temperature
            top_k = preset.top_k
            cfg_musiccoca = preset.cfg_musiccoca
            cfg_notes = preset.cfg_notes
            cfg_drums = preset.cfg_drums
        return self._engine_module.SamplingControls(
            temperature=temperature,
            top_k=top_k,
            cfg_musiccoca=cfg_musiccoca,
            cfg_notes=cfg_notes,
            cfg_drums=cfg_drums,
        )

    def _plan(self, case: RenderCase):
        engine = self._engine
        if (
            engine is None
            or self._engine_module is None
            or self._melody_module is None
        ):
            raise RuntimeError("adapter has not been prepared")
        self._verify_reference(case.style)
        style = engine.embed(case.style.prompt, case.style.audio_reference)
        style_key = engine.style_cache_key(
            case.style.prompt, case.style.audio_reference
        )
        guide = self._melody_module.MelodyGuide(
            f"evaluation-composition:{case.composition_seed}"
        )
        guide.configure(
            station=case.style.station,
            mood=case.style.mood,
            bpm=case.style.bpm,
            groove=case.style.groove,
            intensity=case.style.intensity,
            melody=case.style.melody and case.note_mode.kind == "app_melody",
            drums=case.style.drums,
            strict_drums=case.drum_mode.kind == "strict",
            guide_mode=(
                "guided" if case.note_mode.kind == "app_melody" else "unconstrained"
            ),
        )
        events = guide.event_frames(case.style.mood, case.total_frames)
        sampling = self._sampling(case)
        masked_notes = (-1,) * 128
        onset_notes = list(masked_notes)
        sustain_notes = list(masked_notes)
        if case.note_mode.kind == "constant":
            onset_notes[case.note_mode.midi_note] = 2
            sustain_notes[case.note_mode.midi_note] = 1
        elif case.note_mode.kind not in {"masked", "app_melody"}:
            raise ValueError(f"unsupported note mode: {case.note_mode.kind}")

        frame_controls: list[tuple[tuple[int, ...], int]] = []
        for frame_index, event in enumerate(events):
            if case.note_mode.kind == "app_melody":
                notes = tuple(event.tokens)
            elif case.note_mode.kind == "constant":
                notes = tuple(onset_notes if frame_index == 0 else sustain_notes)
            else:
                notes = masked_notes
            if case.drum_mode.kind in {"planned", "strict"}:
                drum = int(event.drum)
            elif case.drum_mode.kind == "off":
                drum = 0
            elif case.drum_mode.kind == "masked":
                drum = -1
            else:
                raise ValueError(f"unsupported drum mode: {case.drum_mode.kind}")
            frame_controls.append((notes, drum))

        grouped: list[tuple[tuple[int, ...], int, int]] = []
        for notes, drum in frame_controls:
            if grouped and grouped[-1][:2] == (notes, drum):
                old_notes, old_drum, length = grouped[-1]
                grouped[-1] = old_notes, old_drum, length + 1
            else:
                grouped.append((notes, drum, 1))
        plan = [
            self._engine_module.ConditioningRun(
                style=style,
                key=style_key,
                notes=notes,
                drum=drum,
                frames=frames,
                sampling=sampling,
            )
            for notes, drum, frames in grouped
        ]
        plan_metadata = {
            "composition_seed": case.composition_seed,
            "planner_seed": guide.seed,
            "planner_tonic": guide.tonic,
            "station": case.style.station,
            "bpm": case.style.bpm,
            "groove": case.style.groove,
            "intensity": case.style.intensity,
            "melody_enabled": case.style.melody,
            "drums_enabled": case.style.drums,
            "note_mode": case.note_mode.kind,
            "drum_mode": case.drum_mode.kind,
            "effective_sampling": {
                "temperature": sampling.temperature,
                "top_k": sampling.top_k,
                "cfg_musiccoca": sampling.cfg_musiccoca,
                "cfg_notes": sampling.cfg_notes,
                "cfg_drums": sampling.cfg_drums,
            },
        }
        return plan, plan_metadata

    def render(self, case: RenderCase) -> AdapterRender:
        engine = self._engine
        if engine is None:
            raise RuntimeError("adapter has not been prepared")
        engine.pinned_codebooks = case.codebook_depth
        engine.set_codebooks(case.codebook_depth)
        if engine.codebooks != case.codebook_depth:
            raise RuntimeError(
                f"requested {case.codebook_depth} codebooks but engine selected "
                f"{engine.codebooks}"
            )
        plan, plan_metadata = self._plan(case)
        pcm, _state = engine.generate(None, plan, seed=case.seed)
        return AdapterRender(
            pcm_s16le=pcm,
            metadata={
                "engine_class": "MRTEngine",
                "model_size": engine.size,
                "backend": engine.backend,
                "bits": engine.bits,
                "fast_path": engine._fast,
                "fast_sampler": engine._fast_sampling,
                "effective_seed_uint32": int(case.seed) & 0xFFFFFFFF,
                "audio_reference": case.style.audio_reference,
                "audio_reference_sha256": case.style.audio_reference_sha256,
                "audio_style_blend": engine.audio_style_blend,
                "actual_codebooks": engine.codebooks,
                "max_codebooks": engine.max_codebooks,
                "plan": plan_metadata,
            },
        )

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._engine_module = None
        self._melody_module = None


def production_adapter_factory(
    engine_spec: EngineSpec, preset: ParameterPreset
) -> MRTEngineAdapter:
    return MRTEngineAdapter(engine_spec, preset)
