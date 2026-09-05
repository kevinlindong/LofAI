"""Descriptive audio diagnostics for render comparisons.

These measurements catch broken, unexpectedly quiet, clipped, bandwidth-
limited, noise-like, or mechanically repetitive renders.  They intentionally
do not combine into a score: none can decide whether a piece is good music.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Final

import numpy as np


METRICS_DISCLAIMER: Final = (
    "Objective signal diagnostics only. These values are not a music-quality "
    "score or a substitute for blinded human listening."
)
_EPS: Final = np.finfo(np.float64).tiny


@dataclass(frozen=True)
class AudioMetrics:
    duration_seconds: float
    sample_rate: int
    channels: int
    peak_dbfs: float
    rms_dbfs: float
    gated_loudness_proxy_dbfs: float
    clipped_sample_fraction: float
    dc_offset_max_abs: float
    spectral_centroid_hz: float
    spectral_rolloff_95_hz: float
    high_band_energy_fraction: float
    spectral_flatness: float
    onset_proxy_per_second: float
    spectral_flux_mean: float
    repetition_peak_correlation: float
    repetition_peak_lag_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _db(value: float, *, power: bool = False) -> float:
    if value <= 0.0:
        return float("-inf")
    return float((10.0 if power else 20.0) * math.log10(value))


def _as_float_audio(samples: np.ndarray) -> np.ndarray:
    value = np.asarray(samples)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("audio must have shape [frames] or [frames, channels]")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError("audio samples must be numeric")
    if np.issubdtype(value.dtype, np.integer):
        info = np.iinfo(value.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        value = value.astype(np.float64) / scale
    else:
        value = value.astype(np.float64, copy=False)
    if not np.isfinite(value).all():
        raise ValueError("audio samples contain NaN or infinity")
    return value


def _spectral_analysis(
    mono: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Return mean power and frame flux without retaining a full spectrogram."""

    window_size = min(4096, mono.size)
    if window_size < 16:
        padded = np.pad(mono, (0, 16 - mono.size))
        window_size = 16
    else:
        padded = mono
    hop = max(1, window_size // 4)
    starts = list(range(0, max(1, padded.size - window_size + 1), hop))
    final_start = max(0, padded.size - window_size)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    window = np.hanning(window_size)
    power_sum = np.zeros(window_size // 2 + 1, dtype=np.float64)
    previous_normalized = None
    flux: list[float] = []
    for start in starts:
        magnitude = np.abs(
            np.fft.rfft(padded[start : start + window_size] * window)
        )
        power_sum += magnitude**2
        normalized = magnitude / max(float(np.linalg.norm(magnitude)), _EPS)
        if previous_normalized is not None:
            flux.append(float(np.maximum(0.0, normalized - previous_normalized).sum()))
        previous_normalized = normalized
    frequencies = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
    return power_sum / len(starts), frequencies, hop, np.asarray(flux)


def _spectrum_summary(
    mean_power: np.ndarray, frequencies: np.ndarray
) -> tuple[float, float, float, float]:
    audible = (frequencies >= 40.0) & (frequencies <= 20_000.0)
    audible_power = mean_power[audible]
    audible_freq = frequencies[audible]
    total = float(np.sum(audible_power))
    if total <= _EPS or audible_power.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    centroid = float(np.sum(audible_freq * audible_power) / total)
    cumulative = np.cumsum(audible_power)
    rolloff_index = min(
        int(np.searchsorted(cumulative, 0.95 * cumulative[-1])),
        audible_freq.size - 1,
    )
    rolloff = float(audible_freq[rolloff_index])
    high = float(
        np.sum(mean_power[(frequencies >= 8_000.0) & (frequencies <= 20_000.0)])
        / total
    )
    arithmetic = float(np.mean(audible_power))
    floor = max(arithmetic * 1e-12, _EPS)
    flatness = float(
        np.exp(np.mean(np.log(np.maximum(audible_power, floor)))) / arithmetic
    )
    return centroid, rolloff, high, flatness


def _gated_loudness_proxy(audio: np.ndarray, sample_rate: int) -> float:
    """Return an unweighted BS.1770-inspired gated level.

    The 400 ms blocks, overlap, absolute gate, relative gate, and channel power
    sum resemble integrated LUFS, but this intentionally omits K-weighting and
    calibration.  The explicit ``proxy`` name prevents false LUFS precision.
    """

    block_size = max(1, round(sample_rate * 0.4))
    hop = max(1, round(sample_rate * 0.1))
    if audio.shape[0] < block_size:
        audio = np.pad(audio, ((0, block_size - audio.shape[0]), (0, 0)))
    starts = range(0, audio.shape[0] - block_size + 1, hop)
    powers = np.asarray(
        [np.mean(np.sum(audio[start : start + block_size] ** 2, axis=1)) for start in starts],
        dtype=np.float64,
    )
    levels = -0.691 + np.asarray([_db(float(value), power=True) for value in powers])
    absolute = powers[levels > -70.0]
    if absolute.size == 0:
        return float("-inf")
    ungated = -0.691 + _db(float(np.mean(absolute)), power=True)
    relative_threshold = ungated - 10.0
    kept = powers[(levels > -70.0) & (levels > relative_threshold)]
    if kept.size == 0:
        return float("-inf")
    return float(-0.691 + _db(float(np.mean(kept)), power=True))


def _onset_proxy(
    flux: np.ndarray, hop: int, sample_rate: int, duration: float
) -> tuple[float, float]:
    if flux.size < 3 or duration <= 0.0:
        return 0.0, 0.0
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median)))
    threshold = median + max(3.0 * mad, 1e-4)
    candidates = np.flatnonzero(
        (flux[1:-1] > flux[:-2])
        & (flux[1:-1] >= flux[2:])
        & (flux[1:-1] > threshold)
    ) + 1
    min_gap = max(1, round(0.08 * sample_rate / hop))
    accepted: list[int] = []
    for index in candidates:
        if not accepted or index - accepted[-1] >= min_gap:
            accepted.append(int(index))
        elif flux[index] > flux[accepted[-1]]:
            accepted[-1] = int(index)
    return len(accepted) / duration, float(np.mean(flux))


def _repetition_proxy(mono: np.ndarray, sample_rate: int) -> tuple[float, float | None]:
    """Measure periodic similarity of a 50 ms energy envelope.

    This can expose a short stuck loop, but high values are also normal for a
    steady groove.  It is deliberately named a proxy and never thresholded as
    good or bad.
    """

    block = max(1, round(sample_rate * 0.05))
    count = mono.size // block
    if count < 20:
        return 0.0, None
    trimmed = mono[: count * block].reshape(count, block)
    envelope = np.sqrt(np.mean(trimmed**2, axis=1))
    mean_envelope = float(np.mean(envelope))
    envelope -= np.mean(envelope)
    scale = float(np.linalg.norm(envelope))
    if scale <= max(_EPS, mean_envelope * math.sqrt(count) * 1e-6):
        return 0.0, None

    rate = sample_rate / block
    min_lag = max(1, round(0.5 * rate))
    max_lag = min(round(8.0 * rate), count // 2)
    if max_lag < min_lag:
        return 0.0, None
    best = -1.0
    best_lag = min_lag
    for lag in range(min_lag, max_lag + 1):
        left = envelope[:-lag]
        right = envelope[lag:]
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlation = float(np.dot(left, right) / denom) if denom > _EPS else 0.0
        # Prefer the shortest recurrence when numerically equivalent periodic
        # peaks occur at several multiples of the same loop.
        if correlation > best + 1e-9:
            best = correlation
            best_lag = lag
    return max(-1.0, min(1.0, best)), float(best_lag / rate)


def analyze_audio(samples: np.ndarray, sample_rate: int) -> AudioMetrics:
    """Analyze decoded PCM without assigning a musical quality score."""

    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    audio = _as_float_audio(samples)
    duration = audio.shape[0] / sample_rate
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    clipping = float(np.mean(np.abs(audio) >= (32_760.0 / 32_768.0)))
    dc = float(np.max(np.abs(np.mean(audio, axis=0))))
    mono = np.mean(audio, axis=1)
    centered = mono - np.mean(mono)
    mean_power, frequencies, hop, frame_flux = _spectral_analysis(
        centered, sample_rate
    )
    centroid, rolloff, high, flatness = _spectrum_summary(
        mean_power, frequencies
    )
    onset_rate, flux = _onset_proxy(frame_flux, hop, sample_rate, duration)
    repetition, repetition_lag = _repetition_proxy(centered, sample_rate)
    return AudioMetrics(
        duration_seconds=float(duration),
        sample_rate=sample_rate,
        channels=audio.shape[1],
        peak_dbfs=_db(peak),
        rms_dbfs=_db(rms),
        gated_loudness_proxy_dbfs=_gated_loudness_proxy(audio, sample_rate),
        clipped_sample_fraction=clipping,
        dc_offset_max_abs=dc,
        spectral_centroid_hz=centroid,
        spectral_rolloff_95_hz=rolloff,
        high_band_energy_fraction=high,
        spectral_flatness=flatness,
        onset_proxy_per_second=onset_rate,
        spectral_flux_mean=flux,
        repetition_peak_correlation=repetition,
        repetition_peak_lag_seconds=repetition_lag,
    )
