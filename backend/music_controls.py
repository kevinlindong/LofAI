"""Validation and normalization for listener-facing music controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import styles


MIN_BPM = 60
MAX_BPM = 110


def _clamp_number(value: Any, low: float, high: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number:  # NaN
        return fallback
    return max(low, min(high, number))


def _bool(value: Any, fallback: bool) -> bool:
    return value if isinstance(value, bool) else fallback


@dataclass(frozen=True)
class MusicControls:
    station: str
    mood: str
    instrument: str
    bpm: int
    groove: float
    intensity: float
    melody: bool
    drums: bool

    @classmethod
    def initial(
        cls,
        mood: str = styles.DEFAULT_MOOD,
        instrument: str = styles.DEFAULT_INSTRUMENT,
        station: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> "MusicControls":
        # Legacy clients did not send a station; retain their mood/instrument
        # semantics. New clients select a curated station and inherit its grid.
        chosen_station = (
            styles.CUSTOM_STATION
            if station is None
            else styles.normalize_station(station)
        )
        if chosen_station == styles.CUSTOM_STATION:
            mood, instrument = styles.normalize(mood, instrument)
            controls = cls(chosen_station, mood, instrument, 78, 0.60, 0.45, True, True)
        else:
            preset = styles.station_defaults(chosen_station)
            controls = cls(
                chosen_station,
                preset.mood,
                preset.instrument,
                preset.bpm,
                preset.groove,
                preset.intensity,
                True,
                True,
            )
        return controls.update(payload or {}, apply_station_defaults=False)

    def update(
        self,
        payload: Mapping[str, Any],
        *,
        apply_station_defaults: bool = True,
    ) -> "MusicControls":
        next_controls = self
        raw_station = payload.get("station")
        if isinstance(raw_station, str):
            station = styles.normalize_station(raw_station)
            if station != self.station and apply_station_defaults:
                if station == styles.CUSTOM_STATION:
                    next_controls = replace(next_controls, station=station)
                else:
                    preset = styles.station_defaults(station)
                    next_controls = replace(
                        next_controls,
                        station=station,
                        mood=preset.mood,
                        instrument=preset.instrument,
                        bpm=preset.bpm,
                        groove=preset.groove,
                        intensity=preset.intensity,
                    )
            else:
                next_controls = replace(next_controls, station=station)

        mood, instrument = styles.normalize(
            payload.get("mood", next_controls.mood),
            payload.get("instrument", next_controls.instrument),
        )
        # Manually steering either legacy axis makes the result a custom mix,
        # unless the client explicitly included the station in this message.
        station = next_controls.station
        if raw_station is None and (
            "mood" in payload or "instrument" in payload
        ):
            station = styles.CUSTOM_STATION
        elif station != styles.CUSTOM_STATION:
            preset = styles.station_defaults(station)
            contradicts_named_style = (
                "mood" in payload and mood != preset.mood
            ) or (
                "instrument" in payload and instrument != preset.instrument
            )
            if contradicts_named_style:
                # A named prompt has a fixed mood/timbre identity. Do not let a
                # non-UI client silently pair it with a contradictory planner.
                station = styles.CUSTOM_STATION

        return replace(
            next_controls,
            station=station,
            mood=mood,
            instrument=instrument,
            bpm=round(
                _clamp_number(payload.get("bpm"), MIN_BPM, MAX_BPM, next_controls.bpm)
            ),
            groove=_clamp_number(
                payload.get("groove"), 0.0, 1.0, next_controls.groove
            ),
            intensity=_clamp_number(
                payload.get("intensity"), 0.0, 1.0, next_controls.intensity
            ),
            melody=_bool(payload.get("melody"), next_controls.melody),
            drums=_bool(payload.get("drums"), next_controls.drums),
        )

    def prompt(self) -> str:
        return styles.prompt_for(self.mood, self.instrument, self.station)

    def reference(self) -> str | None:
        return styles.audio_reference_for(self.station)

    def payload(self) -> dict[str, Any]:
        return asdict(self)
