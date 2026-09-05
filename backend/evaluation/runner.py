"""Model-agnostic matrix runner and durable manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Protocol, Sequence
import wave

import numpy as np

from .audio_metrics import METRICS_DISCLAIMER, analyze_audio
from .spec import (
    CHANNELS,
    FRAME_RATE,
    SAMPLE_RATE,
    EngineSpec,
    EvaluationSpec,
    ParameterPreset,
    RenderCase,
    StyleSpec,
)


SAMPLES_PER_MODEL_FRAME = SAMPLE_RATE // FRAME_RATE
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AdapterRender:
    """Raw production-path output plus facts observed from the adapter."""

    pcm_s16le: bytes
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    metadata: dict[str, Any] | None = None


class RenderAdapter(Protocol):
    """Small seam that makes the expensive model replaceable in unit tests."""

    def prepare(self, styles: Sequence[StyleSpec]) -> None: ...

    def render(self, case: RenderCase) -> AdapterRender: ...

    def close(self) -> None: ...


AdapterFactory = Callable[[EngineSpec, ParameterPreset], RenderAdapter]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        _safe_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def provenance() -> dict[str, Any]:
    return {
        "generated_at_utc": _utc_now(),
        "git_revision": _git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "magenta-rt": _package_version("magenta-rt"),
            "mlx": _package_version("mlx"),
            "numpy": _package_version("numpy"),
        },
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(_safe_json(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_pcm16_wav(
    path: str | Path, pcm_s16le: bytes, *, sample_rate: int, channels: int
) -> None:
    target = Path(path)
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("sample_rate and channels must be positive")
    frame_width = channels * 2
    if len(pcm_s16le) % frame_width:
        raise ValueError("PCM byte length does not contain complete int16 frames")
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm_s16le)


def read_pcm16_wav(path: str | Path) -> tuple[np.ndarray, int]:
    source = Path(path)
    with wave.open(str(source), "rb") as input_file:
        if input_file.getcomptype() != "NONE" or input_file.getsampwidth() != 2:
            raise ValueError(f"{source} must be uncompressed signed 16-bit PCM WAV")
        channels = input_file.getnchannels()
        sample_rate = input_file.getframerate()
        frames = input_file.getnframes()
        payload = input_file.readframes(frames)
    samples = np.frombuffer(payload, dtype="<i2")
    if channels <= 0 or samples.size % channels:
        raise ValueError(f"{source} has an invalid channel layout")
    return samples.reshape(-1, channels), sample_rate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _new_manifest(spec: EvaluationSpec) -> dict[str, Any]:
    config = spec.to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "lofai_mrt2_evaluation",
        "metrics_disclaimer": METRICS_DISCLAIMER,
        "config_sha256": _canonical_digest(config),
        "config": config,
        "provenance": provenance(),
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "cases": [],
    }


def _load_or_create_manifest(
    path: Path, spec: EvaluationSpec, *, resume: bool
) -> dict[str, Any]:
    if not path.exists():
        return _new_manifest(spec)
    if not resume:
        raise FileExistsError(f"{path} already exists; use --resume or a new output directory")
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = _canonical_digest(spec.to_dict())
    if raw.get("config_sha256") != expected:
        raise ValueError("existing manifest config does not match this evaluation config")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("existing manifest schema version is not supported")
    return raw


_CSV_FIELDS = (
    "case_id",
    "ordinal",
    "status",
    "wav_path",
    "wav_sha256",
    "decoder_seed",
    "composition_seed",
    "style_id",
    "station",
    "prompt",
    "mood",
    "instrument",
    "bpm",
    "groove",
    "intensity",
    "melody",
    "drums",
    "audio_reference",
    "audio_reference_sha256",
    "model_size",
    "backend",
    "bits",
    "fast_sampler_requested",
    "audio_style_blend",
    "preset_id",
    "temperature",
    "top_k",
    "cfg_musiccoca",
    "cfg_notes",
    "cfg_drums",
    "style_token_levels",
    "sampling_mode",
    "note_mode_id",
    "note_mode_kind",
    "midi_note",
    "drum_mode_id",
    "drum_mode_kind",
    "codebook_depth",
    "duration_seconds",
    "warmup_seconds",
    "warmup_bars",
    "warmup_alignment",
    "listening_start_step_in_bar",
    "render_wall_seconds",
    "render_realtime_factor",
    "effective_temperature",
    "effective_top_k",
    "effective_cfg_musiccoca",
    "effective_cfg_notes",
    "effective_cfg_drums",
    "peak_dbfs",
    "rms_dbfs",
    "gated_loudness_proxy_dbfs",
    "clipped_sample_fraction",
    "dc_offset_max_abs",
    "spectral_centroid_hz",
    "spectral_rolloff_95_hz",
    "high_band_energy_fraction",
    "spectral_flatness",
    "onset_proxy_per_second",
    "spectral_flux_mean",
    "repetition_peak_correlation",
    "repetition_peak_lag_seconds",
    "error",
)


def _case_csv(case: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    metrics = case.get("metrics") or {}
    style = case["style"]
    preset = case["preset"]
    note_mode = case["note_mode"]
    drum_mode = case["drum_mode"]
    effective = ((case.get("adapter") or {}).get("plan") or {}).get(
        "effective_sampling", {}
    )
    engine_config = manifest["config"]["engine"]
    row = {
        "case_id": case["case_id"],
        "ordinal": case["ordinal"],
        "status": case["status"],
        "wav_path": case.get("wav_path"),
        "wav_sha256": case.get("wav_sha256"),
        "decoder_seed": case["seed"],
        "composition_seed": case["composition_seed"],
        "style_id": style["id"],
        "station": style["station"],
        "prompt": style["prompt"],
        "mood": style["mood"],
        "instrument": style["instrument"],
        "bpm": style["bpm"],
        "groove": style["groove"],
        "intensity": style["intensity"],
        "melody": style["melody"],
        "drums": style["drums"],
        "audio_reference": style.get("audio_reference"),
        "audio_reference_sha256": style.get("audio_reference_sha256"),
        "model_size": engine_config["model_size"],
        "backend": engine_config["backend"],
        "bits": engine_config["bits"],
        "fast_sampler_requested": engine_config["fast_sampler"],
        "audio_style_blend": engine_config["audio_style_blend"],
        "preset_id": preset["id"],
        "temperature": preset["temperature"],
        "top_k": preset["top_k"],
        "cfg_musiccoca": preset["cfg_musiccoca"],
        "cfg_notes": preset["cfg_notes"],
        "cfg_drums": preset["cfg_drums"],
        "style_token_levels": preset["style_token_levels"],
        "sampling_mode": preset["sampling_mode"],
        "note_mode_id": note_mode["id"],
        "note_mode_kind": note_mode["kind"],
        "midi_note": note_mode.get("midi_note"),
        "drum_mode_id": drum_mode["id"],
        "drum_mode_kind": drum_mode["kind"],
        "codebook_depth": case["codebook_depth"],
        "duration_seconds": case["duration_frames"] / FRAME_RATE,
        "warmup_seconds": case["warmup_frames"] / FRAME_RATE,
        "warmup_bars": case.get("warmup_bars"),
        "warmup_alignment": "bar" if case.get("warmup_bars") is not None else "seconds",
        "listening_start_step_in_bar": case["listening_start_step_in_bar"],
        "render_wall_seconds": case.get("render_wall_seconds"),
        "render_realtime_factor": case.get("render_realtime_factor"),
        "effective_temperature": effective.get("temperature"),
        "effective_top_k": effective.get("top_k"),
        "effective_cfg_musiccoca": effective.get("cfg_musiccoca"),
        "effective_cfg_notes": effective.get("cfg_notes"),
        "effective_cfg_drums": effective.get("cfg_drums"),
        "error": case.get("error"),
    }
    row.update({field: metrics.get(field) for field in AudioMetricFields})
    return row


AudioMetricFields = _CSV_FIELDS[_CSV_FIELDS.index("peak_dbfs") : -1]


def write_manifest_files(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["cases"] = sorted(manifest["cases"], key=lambda value: value["ordinal"])
    manifest["updated_at_utc"] = _utc_now()
    _atomic_json(output_dir / "manifest.json", manifest)
    csv_path = output_dir / "manifest.csv"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output_dir, prefix=".manifest.csv.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for case in manifest["cases"]:
            writer.writerow(_safe_json(_case_csv(case, manifest)))
    temporary.replace(csv_path)


def _completed_case_is_valid(output_dir: Path, existing: dict[str, Any]) -> bool:
    if existing.get("status") != "complete" or not existing.get("wav_path"):
        return False
    path = output_dir / existing["wav_path"]
    return path.is_file() and _sha256_file(path) == existing.get("wav_sha256")


def _case_record(case: RenderCase) -> dict[str, Any]:
    value = case.to_dict()
    value["style"] = asdict(case.style)
    value["preset"] = asdict(case.preset)
    value["note_mode"] = asdict(case.note_mode)
    return value


def _render_one(
    adapter: RenderAdapter, case: RenderCase, output_dir: Path
) -> dict[str, Any]:
    started = time.monotonic()
    rendered = adapter.render(case)
    elapsed = time.monotonic() - started
    if rendered.sample_rate != SAMPLE_RATE or rendered.channels != CHANNELS:
        raise ValueError(
            f"adapter returned {rendered.sample_rate} Hz/{rendered.channels} channels; "
            f"MRT2 evaluation requires {SAMPLE_RATE} Hz/{CHANNELS} channels"
        )
    expected_bytes = case.total_frames * SAMPLES_PER_MODEL_FRAME * CHANNELS * 2
    if len(rendered.pcm_s16le) != expected_bytes:
        raise ValueError(
            f"adapter returned {len(rendered.pcm_s16le)} PCM bytes, expected {expected_bytes}"
        )

    warmup_bytes = case.warmup_frames * SAMPLES_PER_MODEL_FRAME * CHANNELS * 2
    evaluated_pcm = rendered.pcm_s16le[warmup_bytes:]
    wav_relative = Path("wav") / f"{case.case_id}.wav"
    wav_path = output_dir / wav_relative
    write_pcm16_wav(
        wav_path,
        evaluated_pcm,
        sample_rate=rendered.sample_rate,
        channels=rendered.channels,
    )
    samples = np.frombuffer(evaluated_pcm, dtype="<i2").reshape(-1, CHANNELS)
    metrics = analyze_audio(samples, SAMPLE_RATE).to_dict()
    record = _case_record(case)
    record.update(
        {
            "status": "complete",
            "wav_path": wav_relative.as_posix(),
            "wav_sha256": _sha256_file(wav_path),
            "render_wall_seconds": elapsed,
            "render_realtime_factor": (
                case.total_frames / FRAME_RATE / elapsed if elapsed > 0 else None
            ),
            "adapter": rendered.metadata or {},
            "metrics": metrics,
        }
    )
    return record


def run_evaluation(
    spec: EvaluationSpec,
    output_dir: str | Path,
    adapter_factory: AdapterFactory,
    *,
    resume: bool = False,
    continue_on_error: bool = False,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Render every matrix case, checkpointing JSON and CSV after each clip."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    manifest = _load_or_create_manifest(manifest_path, spec, resume=resume)
    by_id = {value["case_id"]: value for value in manifest["cases"]}
    all_cases = spec.expand()
    # Persist the exact config before a potentially minutes-long model load so
    # an interrupted first run still has an auditable, resumable run record.
    write_manifest_files(target, manifest)

    for preset in spec.presets:
        pending = [
            case
            for case in all_cases
            if case.preset.id == preset.id
            and not (
                case.case_id in by_id
                and _completed_case_is_valid(target, by_id[case.case_id])
            )
        ]
        if not pending:
            continue
        adapter = adapter_factory(spec.engine, preset)
        try:
            unique_styles = tuple(
                {case.style.id: case.style for case in pending}.values()
            )
            if progress:
                progress(f"preparing preset {preset.id} ({len(pending)} render(s))")
            adapter.prepare(unique_styles)
            for case in pending:
                if progress:
                    progress(f"[{case.ordinal}/{len(all_cases)}] rendering {case.case_id}")
                try:
                    record = _render_one(adapter, case, target)
                except Exception as exc:
                    record = _case_record(case)
                    record.update(
                        {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    by_id[case.case_id] = record
                    manifest["cases"] = list(by_id.values())
                    write_manifest_files(target, manifest)
                    if not continue_on_error:
                        raise
                else:
                    by_id[case.case_id] = record
                    manifest["cases"] = list(by_id.values())
                    write_manifest_files(target, manifest)
        finally:
            adapter.close()

    manifest["cases"] = list(by_id.values())
    write_manifest_files(target, manifest)
    return manifest
