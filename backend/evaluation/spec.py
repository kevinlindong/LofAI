"""Configuration schema and deterministic matrix expansion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


FRAME_RATE = 25
SAMPLE_RATE = 48_000
CHANNELS = 2


def _require_keys(value: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} field(s): {', '.join(unknown)}")


def _safe_id(value: str, context: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            f"{context} must start with an alphanumeric character and contain "
            "only letters, numbers, '.', '_' or '-': {value!r}"
        )
    return value


def _frames(seconds: float, context: str) -> int:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        raise ValueError(f"{context} must be a non-negative number")
    value = float(seconds) * FRAME_RATE
    rounded = round(value)
    if abs(value - rounded) > 1e-7:
        raise ValueError(f"{context} must be a multiple of 0.04 seconds")
    return int(rounded)


def _project_styles():
    """Import the lightweight live style registry from either supported cwd."""

    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    return importlib.import_module("styles")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EngineSpec:
    model_size: str = "mrt2_small"
    backend: str = "python"
    bits: int = 8
    fast_sampler: bool = True
    audio_style_blend: float = 0.75

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EngineSpec":
        _require_keys(
            raw,
            {
                "model_size",
                "backend",
                "bits",
                "fast_sampler",
                "audio_style_blend",
            },
            "engine",
        )
        result = cls(**raw)
        if result.model_size not in {"mrt2_small", "mrt2_base"}:
            raise ValueError("engine.model_size must be mrt2_small or mrt2_base")
        if result.backend != "python":
            raise ValueError(
                "engine.backend must be python; the MLXFN path is disabled for "
                "quality and reproducible codebook-depth evaluation"
            )
        if result.bits not in {0, 4, 8}:
            raise ValueError("engine.bits must be 0 (full precision), 4, or 8")
        if not isinstance(result.fast_sampler, bool):
            raise ValueError("engine.fast_sampler must be true or false")
        if not 0.0 <= result.audio_style_blend <= 1.0:
            raise ValueError("engine.audio_style_blend must be in [0, 1]")
        return result


@dataclass(frozen=True)
class StyleSpec:
    id: str
    prompt: str
    station: str
    mood: str
    instrument: str
    bpm: int
    groove: float
    intensity: float
    melody: bool
    drums: bool
    audio_reference: str | None = None
    audio_reference_sha256: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StyleSpec":
        _require_keys(
            raw,
            {
                "id",
                "prompt",
                "station",
                "mood",
                "instrument",
                "bpm",
                "groove",
                "intensity",
                "melody",
                "drums",
                "audio_reference",
            },
            "style",
        )
        style_id = _safe_id(raw.get("id"), "style.id")
        project_styles = _project_styles()
        station_was_given = "station" in raw
        station = str(raw.get("station", project_styles.CUSTOM_STATION)).strip()
        if station not in {*project_styles.STATIONS, project_styles.CUSTOM_STATION}:
            raise ValueError(f"style {style_id!r} has unknown station {station!r}")

        if station == project_styles.CUSTOM_STATION:
            defaults = {
                "mood": project_styles.DEFAULT_MOOD,
                "instrument": project_styles.DEFAULT_INSTRUMENT,
                "bpm": 78,
                "groove": 0.60,
                "intensity": 0.45,
            }
        else:
            station_defaults = project_styles.station_defaults(station)
            defaults = {
                "mood": station_defaults.mood,
                "instrument": station_defaults.instrument,
                "bpm": station_defaults.bpm,
                "groove": station_defaults.groove,
                "intensity": station_defaults.intensity,
            }
        mood = raw.get("mood", defaults["mood"])
        instrument = raw.get("instrument", defaults["instrument"])
        if mood not in project_styles.MOODS:
            raise ValueError(f"style {style_id!r} mood must be somber, neutral, or lively")
        if instrument not in project_styles.INSTRUMENTS:
            raise ValueError(
                f"style {style_id!r} instrument must be piano, guitar, or brass"
            )
        if station_was_given and station != project_styles.CUSTOM_STATION:
            live_prompt = project_styles.prompt_for(mood, instrument, station)
            supplied_prompt = raw.get("prompt")
            if supplied_prompt is not None and supplied_prompt != live_prompt:
                raise ValueError(
                    f"style {style_id!r} prompt differs from live station {station!r}; "
                    "omit prompt to resolve the production style"
                )
            prompt = live_prompt
        else:
            prompt = raw.get("prompt") or project_styles.prompt_for(
                mood, instrument, project_styles.CUSTOM_STATION
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"style {style_id!r} has an empty prompt")

        bpm = raw.get("bpm", defaults["bpm"])
        groove = raw.get("groove", defaults["groove"])
        intensity = raw.get("intensity", defaults["intensity"])
        if not isinstance(bpm, (int, float)) or not 60 <= bpm <= 110:
            raise ValueError(f"style {style_id!r} bpm must be in [60, 110]")
        if not isinstance(groove, (int, float)) or not 0.0 <= groove <= 1.0:
            raise ValueError(f"style {style_id!r} groove must be in [0, 1]")
        if not isinstance(intensity, (int, float)) or not 0.0 <= intensity <= 1.0:
            raise ValueError(f"style {style_id!r} intensity must be in [0, 1]")
        melody = raw.get("melody", True)
        drums = raw.get("drums", True)
        if not isinstance(melody, bool) or not isinstance(drums, bool):
            raise ValueError(f"style {style_id!r} melody and drums must be booleans")

        if "audio_reference" in raw:
            reference = raw["audio_reference"]
        else:
            reference = project_styles.audio_reference_for(station)
        reference_path: str | None = None
        reference_sha256: str | None = None
        if reference is not None:
            path = Path(reference).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"style {style_id!r} audio reference does not exist: {path}")
            reference_path = str(path)
            reference_sha256 = _sha256_path(path)
        return cls(
            id=style_id,
            prompt=prompt.strip(),
            station=station,
            mood=mood,
            instrument=instrument,
            bpm=round(float(bpm)),
            groove=float(groove),
            intensity=float(intensity),
            melody=melody,
            drums=drums,
            audio_reference=reference_path,
            audio_reference_sha256=reference_sha256,
        )


@dataclass(frozen=True)
class ParameterPreset:
    id: str
    temperature: float
    top_k: int
    cfg_musiccoca: float
    cfg_notes: float
    cfg_drums: float
    style_token_levels: int = 6
    sampling_mode: str = "raw"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ParameterPreset":
        _require_keys(
            raw,
            {
                "id",
                "temperature",
                "top_k",
                "cfg_musiccoca",
                "cfg_notes",
                "cfg_drums",
                "style_token_levels",
                "sampling_mode",
            },
            "preset",
        )
        result = cls(**raw)
        _safe_id(result.id, "preset.id")
        if result.temperature <= 0:
            raise ValueError(f"preset {result.id!r} temperature must be > 0")
        if not 0 <= result.top_k <= 1024:
            raise ValueError(f"preset {result.id!r} top_k must be in [0, 1024]")
        for key in ("cfg_musiccoca", "cfg_notes", "cfg_drums"):
            value = getattr(result, key)
            if not -1.0 <= value <= 7.0:
                raise ValueError(f"preset {result.id!r} {key} must be in [-1, 7]")
        if not 1 <= result.style_token_levels <= 12:
            raise ValueError(f"preset {result.id!r} style_token_levels must be in [1, 12]")
        if result.sampling_mode not in {"raw", "production"}:
            raise ValueError(
                f"preset {result.id!r} sampling_mode must be raw or production"
            )
        return result


@dataclass(frozen=True)
class NoteMode:
    id: str
    kind: str
    midi_note: int | None = None

    @classmethod
    def from_value(cls, raw: str | dict[str, Any]) -> "NoteMode":
        if isinstance(raw, str):
            raw = {"id": raw, "kind": raw}
        if not isinstance(raw, dict):
            raise ValueError("each note mode must be a string or object")
        _require_keys(raw, {"id", "kind", "midi_note"}, "note mode")
        result = cls(**raw)
        _safe_id(result.id, "note_mode.id")
        if result.kind not in {"masked", "app_melody", "constant"}:
            raise ValueError(
                f"note mode {result.id!r} kind must be masked, app_melody, or constant"
            )
        if result.kind == "constant":
            if result.midi_note is None or not 0 <= result.midi_note <= 127:
                raise ValueError(
                    f"constant note mode {result.id!r} needs midi_note in [0, 127]"
                )
        elif result.midi_note is not None:
            raise ValueError(f"note mode {result.id!r} only uses midi_note for kind=constant")
        return result


@dataclass(frozen=True)
class DrumMode:
    id: str
    kind: str

    @classmethod
    def from_value(cls, raw: str | dict[str, Any]) -> "DrumMode":
        if isinstance(raw, str):
            raw = {"id": raw, "kind": raw}
        if not isinstance(raw, dict):
            raise ValueError("each drum mode must be a string or object")
        _require_keys(raw, {"id", "kind"}, "drum mode")
        result = cls(**raw)
        _safe_id(result.id, "drum_mode.id")
        if result.kind not in {"planned", "strict", "masked", "off"}:
            raise ValueError(
                f"drum mode {result.id!r} kind must be planned, strict, masked, or off"
            )
        return result


@dataclass(frozen=True)
class ReportSpec:
    seed: int = 20260903
    max_pairs: int = 80

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReportSpec":
        _require_keys(raw, {"seed", "max_pairs"}, "report")
        result = cls(**raw)
        if result.max_pairs < 0:
            raise ValueError("report.max_pairs must be >= 0 (0 means no limit)")
        return result


@dataclass(frozen=True)
class EvaluationSpec:
    duration_seconds: float
    warmup_seconds: float | None
    warmup_bars: int | None
    engine: EngineSpec
    seeds: tuple[int, ...]
    composition_seed: int | None
    styles: tuple[StyleSpec, ...]
    presets: tuple[ParameterPreset, ...]
    note_modes: tuple[NoteMode, ...]
    drum_modes: tuple[DrumMode, ...]
    codebook_depths: tuple[int, ...]
    report: ReportSpec

    @property
    def duration_frames(self) -> int:
        return _frames(self.duration_seconds, "duration_seconds")

    def warmup_frames_for(self, style: StyleSpec) -> int:
        if self.warmup_bars is not None:
            exact = self.warmup_bars * 4.0 * 60.0 * FRAME_RATE / style.bpm
            return int(math.ceil(exact - 1e-12))
        return _frames(self.warmup_seconds or 0.0, "warmup_seconds")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["seeds"] = list(self.seeds)
        value["styles"] = [asdict(item) for item in self.styles]
        value["presets"] = [asdict(item) for item in self.presets]
        value["note_modes"] = [asdict(item) for item in self.note_modes]
        value["drum_modes"] = [asdict(item) for item in self.drum_modes]
        value["codebook_depths"] = list(self.codebook_depths)
        return value

    def expand(self) -> list["RenderCase"]:
        cases: list[RenderCase] = []
        combinations = itertools.product(
            self.seeds,
            self.styles,
            self.presets,
            self.note_modes,
            self.drum_modes,
            self.codebook_depths,
        )
        for ordinal, (
            seed,
            style,
            preset,
            note_mode,
            drum_mode,
            codebooks,
        ) in enumerate(
            combinations, start=1
        ):
            composition_seed = self.composition_seed
            if composition_seed is None:
                composition_seed = seed
            warmup_frames = self.warmup_frames_for(style)
            start_step = (
                warmup_frames * style.bpm * 4.0 / (FRAME_RATE * 60.0)
            ) % 16.0
            identity = {
                "decoder_seed": seed,
                "composition_seed": composition_seed,
                "style": asdict(style),
                "preset": asdict(preset),
                "note_mode": asdict(note_mode),
                "drum_mode": asdict(drum_mode),
                "codebook_depth": codebooks,
                "duration_frames": self.duration_frames,
                "warmup_frames": warmup_frames,
                "warmup_bars": self.warmup_bars,
                "engine": asdict(self.engine),
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:10]
            stem = (
                f"{ordinal:04d}__{style.id}__{preset.id}__{note_mode.id}"
                f"__{drum_mode.id}__cb{codebooks}__s{seed}__{digest}"
            )
            cases.append(
                RenderCase(
                    ordinal=ordinal,
                    case_id=stem,
                    seed=seed,
                    composition_seed=composition_seed,
                    style=style,
                    preset=preset,
                    note_mode=note_mode,
                    drum_mode=drum_mode,
                    codebook_depth=codebooks,
                    duration_frames=self.duration_frames,
                    warmup_frames=warmup_frames,
                    warmup_bars=self.warmup_bars,
                    listening_start_step_in_bar=start_step,
                )
            )
        return cases


@dataclass(frozen=True)
class RenderCase:
    ordinal: int
    case_id: str
    seed: int
    composition_seed: int
    style: StyleSpec
    preset: ParameterPreset
    note_mode: NoteMode
    drum_mode: DrumMode
    codebook_depth: int
    duration_frames: int
    warmup_frames: int
    warmup_bars: int | None
    listening_start_step_in_bar: float

    @property
    def total_frames(self) -> int:
        return self.duration_frames + self.warmup_frames

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique_ids(values: tuple[Any, ...], context: str) -> None:
    ids = [value.id for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context} ids must be unique")


def evaluation_spec_from_dict(raw: dict[str, Any]) -> EvaluationSpec:
    _require_keys(
        raw,
        {
            "schema_version",
            "duration_seconds",
            "warmup_seconds",
            "warmup_bars",
            "engine",
            "seeds",
            "composition_seed",
            "styles",
            "presets",
            "note_modes",
            "drum_modes",
            "codebook_depths",
            "report",
        },
        "top-level",
    )
    if raw.get("schema_version", 1) != 1:
        raise ValueError("only evaluation schema_version 1 is supported")
    if "warmup_seconds" in raw and "warmup_bars" in raw:
        raise ValueError("use either warmup_seconds or warmup_bars, not both")
    warmup_bars = raw.get("warmup_bars")
    warmup_seconds = None if warmup_bars is not None else raw.get("warmup_seconds", 2.0)
    if warmup_bars is not None and (
        not isinstance(warmup_bars, int) or isinstance(warmup_bars, bool) or warmup_bars < 0
    ):
        raise ValueError("warmup_bars must be a non-negative integer")
    composition_seed = raw.get("composition_seed")
    if composition_seed is not None and not isinstance(composition_seed, int):
        raise ValueError("composition_seed must be an integer or null")
    spec = EvaluationSpec(
        duration_seconds=raw.get("duration_seconds", 20.0),
        warmup_seconds=warmup_seconds,
        warmup_bars=warmup_bars,
        engine=EngineSpec.from_dict(raw.get("engine", {})),
        seeds=tuple(raw.get("seeds", (20260903,))),
        composition_seed=composition_seed,
        styles=tuple(StyleSpec.from_dict(value) for value in raw.get("styles", ())),
        presets=tuple(
            ParameterPreset.from_dict(value) for value in raw.get("presets", ())
        ),
        note_modes=tuple(
            NoteMode.from_value(value) for value in raw.get("note_modes", ("masked",))
        ),
        drum_modes=tuple(
            DrumMode.from_value(value) for value in raw.get("drum_modes", ("planned",))
        ),
        codebook_depths=tuple(raw.get("codebook_depths", (12,))),
        report=ReportSpec.from_dict(raw.get("report", {})),
    )
    if spec.duration_frames <= 0:
        raise ValueError("duration_seconds must contain at least one 40 ms frame")
    if not spec.seeds or not all(isinstance(seed, int) for seed in spec.seeds):
        raise ValueError("seeds must be a non-empty list of integers")
    if not spec.styles:
        raise ValueError("styles must contain at least one style")
    if not spec.presets:
        raise ValueError("presets must contain at least one preset")
    if not spec.note_modes:
        raise ValueError("note_modes must contain at least one mode")
    if not spec.drum_modes:
        raise ValueError("drum_modes must contain at least one mode")
    if not spec.codebook_depths or not all(
        isinstance(depth, int) and 8 <= depth <= 12 for depth in spec.codebook_depths
    ):
        raise ValueError("codebook_depths must be a non-empty list of integers in [8, 12]")
    if len(set(spec.seeds)) != len(spec.seeds):
        raise ValueError("seeds must be unique")
    if len(set(spec.codebook_depths)) != len(spec.codebook_depths):
        raise ValueError("codebook_depths must be unique")
    _unique_ids(spec.styles, "style")
    _unique_ids(spec.presets, "preset")
    _unique_ids(spec.note_modes, "note mode")
    _unique_ids(spec.drum_modes, "drum mode")
    return spec


def load_evaluation_spec(path: str | Path) -> EvaluationSpec:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("evaluation config must be a JSON object")
    return evaluation_spec_from_dict(raw)
