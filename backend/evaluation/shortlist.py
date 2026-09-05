"""Conservative signal-risk triage for large multi-take batches."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .runner import _atomic_json


SHORTLIST_DISCLAIMER = (
    "Non-authoritative heuristic shortlist. It only deprioritizes possible PCM "
    "pathologies using broad signal diagnostics; it cannot assess melody, harmony, "
    "groove, structure, originality, emotion, or whether anyone likes the music. "
    "Use blinded listening for the decision."
)


def _number(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return default


def signal_risk(case: dict[str, Any]) -> tuple[float, list[str]]:
    """Return broad risk points and human-readable flags, never a quality score."""

    metrics = case.get("metrics") or {}
    flags: list[str] = []
    points = 0.0

    clipped = _number(metrics, "clipped_sample_fraction")
    if clipped > 0.005:
        flags.append("more than 0.5% of samples are near the digital rails")
        points += 10.0 + clipped * 100.0

    rms = _number(metrics, "rms_dbfs", -100.0)
    if rms < -45.0:
        flags.append("unusually low RMS level")
        points += min(10.0, (-45.0 - rms) / 4.0)
    elif rms > -3.0:
        flags.append("unusually high sustained RMS level")
        points += min(10.0, (rms + 3.0) * 2.0)

    flatness = _number(metrics, "spectral_flatness")
    high_band = _number(metrics, "high_band_energy_fraction")
    if flatness > 0.65 and high_band > 0.60:
        flags.append("combined flat/high-band signature may be broadband noise")
        points += (flatness - 0.65) * 10.0 + (high_band - 0.60) * 10.0

    rolloff = _number(metrics, "spectral_rolloff_95_hz")
    if 0.0 < rolloff < 2_500.0:
        flags.append("95% spectral rolloff is below 2.5 kHz")
        points += min(5.0, (2_500.0 - rolloff) / 500.0)

    repetition = _number(metrics, "repetition_peak_correlation")
    if repetition > 0.995:
        flags.append("energy envelope has an almost exact short recurrence")
        points += min(3.0, (repetition - 0.995) * 200.0)

    return round(points, 6), flags


def build_signal_shortlist(
    manifest_path: str | Path,
    *,
    count: int = 3,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("shortlist count must be positive")
    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    candidates = []
    for case in manifest.get("cases", []):
        if case.get("status") != "complete":
            continue
        points, flags = signal_risk(case)
        candidates.append(
            {
                "case_id": case["case_id"],
                "wav_path": case.get("wav_path"),
                "decoder_seed": case["seed"],
                "composition_seed": case["composition_seed"],
                "signal_risk_points": points,
                "signal_risk_flags": flags,
            }
        )
    if not candidates:
        raise ValueError("manifest contains no completed renders")
    candidates.sort(
        key=lambda value: (
            len(value["signal_risk_flags"]),
            value["signal_risk_points"],
            value["case_id"],
        )
    )
    selected = candidates[: min(count, len(candidates))]
    for rank, item in enumerate(selected, start=1):
        item["triage_rank"] = rank
    result = {
        "schema_version": 1,
        "kind": "lofai_non_authoritative_signal_shortlist",
        "disclaimer": SHORTLIST_DISCLAIMER,
        "method": (
            "Prefer fewer broad signal-risk flags, then fewer risk points, then "
            "stable case ID order. No musical feature is rewarded."
        ),
        "source_manifest": source.name,
        "completed_candidate_count": len(candidates),
        "shortlist": selected,
    }
    target = Path(output_path) if output_path else source.parent / "signal_shortlist.json"
    _atomic_json(target, result)
    return result
