"""Conservative startup checks for 48 kHz stereo signed-int16 PCM.

This module intentionally detects only hard failures.  It is not a music
quality score: quiet masters, tape hiss, distortion, and bright lo-fi mixes
should all have ample room inside the thresholds below.  In particular, the
white-noise check requires several independent noise signatures at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


EXPECTED_SAMPLE_RATE: Final = 48_000
EXPECTED_CHANNELS: Final = 2
MIN_DURATION_SECONDS: Final = 0.5

# Broad hard-failure boundaries.  These are deliberately not mastering
# targets: normal music should sit comfortably between them.
MIN_RMS_DBFS: Final = -55.0
MAX_RMS_DBFS: Final = -2.5
MAX_ABS_DC_OFFSET: Final = 0.08
MAX_CLIPPED_FRACTION: Final = 0.03

# A failure needs all of these signatures.  A single metric is too easy for
# legitimate percussion, tape hiss, or a bright arrangement to trip.
MIN_NOISE_ZCR: Final = 0.30
MIN_NOISE_FLATNESS: Final = 0.55
MAX_NOISE_LOW_BAND_FRACTION: Final = 0.08
MIN_NOISE_HIGH_BAND_FRACTION: Final = 0.65

_CLIP_LEVEL: Final = 32_760.0 / 32_768.0
_ISSUE_TEXT: Final = {
    "invalid_type": "PCM must be signed-int16 bytes or a NumPy array",
    "invalid_sample_rate": "PCM is not 48 kHz",
    "invalid_channel_count": "PCM is not stereo",
    "invalid_byte_length": "PCM byte length is not aligned to int16 samples",
    "invalid_frame_layout": "PCM samples do not form complete stereo frames",
    "invalid_sample_dtype": "PCM array is not signed int16",
    "empty": "PCM contains no audio frames",
    "too_short": "PCM is too short for a reliable startup quality check",
    "silence": "PCM is silent or effectively silent",
    "rms_too_high": "PCM has an extreme sustained level",
    "dc_offset": "PCM has an extreme DC offset",
    "clipping": "PCM has sustained digital clipping",
    "white_noise": "PCM has a broadband white-noise signature",
}


@dataclass(frozen=True)
class PCMQualityReport:
    """Measurements and stable issue codes from one PCM quality check."""

    issues: tuple[str, ...]
    sample_rate: int
    channels: int
    frame_count: int
    duration_seconds: float
    rms_dbfs: float | None = None
    peak_dbfs: float | None = None
    max_abs_dc_offset: float | None = None
    clipped_fraction: float | None = None
    zero_crossing_rate: float | None = None
    spectral_flatness: float | None = None
    low_band_fraction: float | None = None
    mid_band_fraction: float | None = None
    high_band_fraction: float | None = None

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def ok(self) -> bool:
        """Alias that reads naturally at call sites."""

        return self.passed

    def summary(self) -> str:
        if self.passed:
            return "startup PCM passed quality checks"
        return "; ".join(_ISSUE_TEXT.get(issue, issue) for issue in self.issues)


class PCMQualityError(RuntimeError):
    """Raised when startup PCM has a hard quality failure."""

    def __init__(self, report: PCMQualityReport):
        self.report = report
        super().__init__(f"startup PCM quality check failed: {report.summary()}")


def _structural_report(
    issues: list[str], sample_rate: int, channels: int
) -> PCMQualityReport:
    return PCMQualityReport(
        issues=tuple(dict.fromkeys(issues)),
        sample_rate=sample_rate,
        channels=channels,
        frame_count=0,
        duration_seconds=0.0,
    )


def _decode_pcm(
    pcm: bytes | bytearray | memoryview | np.ndarray,
    channels: int,
    issues: list[str],
) -> np.ndarray | None:
    """Return a ``[frames, channels]`` int16 view, or record why it cannot."""

    if channels <= 0:
        return None

    if isinstance(pcm, (bytes, bytearray, memoryview)):
        raw = bytes(pcm)
        if len(raw) % np.dtype("<i2").itemsize:
            issues.append("invalid_byte_length")
            return None
        samples = np.frombuffer(raw, dtype="<i2")
        if samples.size % channels:
            issues.append("invalid_frame_layout")
            return None
        return samples.reshape(-1, channels)

    if not isinstance(pcm, np.ndarray):
        issues.append("invalid_type")
        return None
    if pcm.dtype.kind != "i" or pcm.dtype.itemsize != 2:
        issues.append("invalid_sample_dtype")
        return None

    if pcm.ndim == 1:
        if pcm.size % channels:
            issues.append("invalid_frame_layout")
            return None
        return pcm.astype(np.int16, copy=False).reshape(-1, channels)
    if pcm.ndim == 2 and pcm.shape[1] == channels:
        return pcm.astype(np.int16, copy=False)

    issues.append("invalid_frame_layout")
    return None


def _dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(amplitude))


def _zero_crossing_rate(centered: np.ndarray) -> float:
    if centered.shape[0] < 2:
        return 0.0
    # Multiplication avoids counting exact zero plateaus as a flurry of
    # crossings.  DC removal makes the metric insensitive to ordinary offset.
    crossings = centered[1:] * centered[:-1] < 0.0
    return float(np.mean(crossings))


def _spectrum_metrics(
    centered: np.ndarray, sample_rate: int
) -> tuple[float, float, float, float]:
    """Return flatness and low/mid/high audible-band energy fractions."""

    frame_count = centered.shape[0]
    if frame_count < 2:
        return 0.0, 0.0, 0.0, 0.0

    segment_size = min(4096, frame_count)
    hop = max(1, segment_size // 2)
    window = np.hanning(segment_size).astype(np.float64)
    spectrum_sum = np.zeros(segment_size // 2 + 1, dtype=np.float64)
    segment_count = 0

    final_start = frame_count - segment_size
    starts = range(0, final_start + 1, hop)
    for start in starts:
        block = centered[start : start + segment_size] * window[:, None]
        spectrum = np.fft.rfft(block, axis=0)
        spectrum_sum += np.mean(np.abs(spectrum) ** 2, axis=1)
        segment_count += 1

    if segment_count == 0:
        return 0.0, 0.0, 0.0, 0.0

    power = spectrum_sum / segment_count
    frequencies = np.fft.rfftfreq(segment_size, d=1.0 / sample_rate)
    audible = (frequencies >= 40.0) & (frequencies < 20_000.0)
    audible_power = power[audible]
    total = float(np.sum(audible_power))
    if total <= np.finfo(np.float64).tiny:
        return 0.0, 0.0, 0.0, 0.0

    arithmetic_mean = float(np.mean(audible_power))
    floor = max(arithmetic_mean * 1e-12, np.finfo(np.float64).tiny)
    flatness = float(
        np.exp(np.mean(np.log(np.maximum(audible_power, floor))))
        / arithmetic_mean
    )

    low = float(np.sum(power[(frequencies >= 40.0) & (frequencies < 500.0)]))
    mid = float(np.sum(power[(frequencies >= 500.0) & (frequencies < 4_000.0)]))
    high = float(
        np.sum(power[(frequencies >= 4_000.0) & (frequencies < 20_000.0)])
    )
    return flatness, low / total, mid / total, high / total


def analyze_startup_pcm(
    pcm: bytes | bytearray | memoryview | np.ndarray,
    *,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
    channels: int = EXPECTED_CHANNELS,
) -> PCMQualityReport:
    """Analyze one startup PCM buffer without mutating it or external state."""

    issues: list[str] = []
    if sample_rate != EXPECTED_SAMPLE_RATE:
        issues.append("invalid_sample_rate")
    if channels != EXPECTED_CHANNELS:
        issues.append("invalid_channel_count")

    samples = _decode_pcm(pcm, channels, issues)
    if samples is None:
        return _structural_report(issues, sample_rate, channels)

    frame_count = int(samples.shape[0])
    duration = frame_count / sample_rate if sample_rate > 0 else 0.0
    if frame_count == 0:
        issues.append("empty")
        return PCMQualityReport(
            issues=tuple(dict.fromkeys(issues)),
            sample_rate=sample_rate,
            channels=channels,
            frame_count=0,
            duration_seconds=duration,
        )
    if sample_rate <= 0:
        # The invalid sample-rate issue above is enough to reject the buffer;
        # avoid asking FFT helpers to interpret nonsensical timing metadata.
        return PCMQualityReport(
            issues=tuple(dict.fromkeys(issues)),
            sample_rate=sample_rate,
            channels=channels,
            frame_count=frame_count,
            duration_seconds=0.0,
        )
    if duration < MIN_DURATION_SECONDS:
        issues.append("too_short")

    normalized = samples.astype(np.float64) / 32_768.0
    rms = float(np.sqrt(np.mean(normalized**2)))
    peak = float(np.max(np.abs(normalized)))
    dc = float(np.max(np.abs(np.mean(normalized, axis=0))))
    clipped = float(np.mean(np.abs(normalized) >= _CLIP_LEVEL))
    centered = normalized - np.mean(normalized, axis=0, keepdims=True)
    zcr = _zero_crossing_rate(centered)
    flatness, low, mid, high = _spectrum_metrics(centered, sample_rate)
    rms_dbfs = _dbfs(rms)
    peak_dbfs = _dbfs(peak)

    if rms_dbfs < MIN_RMS_DBFS:
        issues.append("silence")
    if rms_dbfs > MAX_RMS_DBFS:
        issues.append("rms_too_high")
    if dc > MAX_ABS_DC_OFFSET:
        issues.append("dc_offset")
    if clipped > MAX_CLIPPED_FRACTION:
        issues.append("clipping")
    if (
        zcr > MIN_NOISE_ZCR
        and flatness > MIN_NOISE_FLATNESS
        and low < MAX_NOISE_LOW_BAND_FRACTION
        and high > MIN_NOISE_HIGH_BAND_FRACTION
    ):
        issues.append("white_noise")

    return PCMQualityReport(
        issues=tuple(dict.fromkeys(issues)),
        sample_rate=sample_rate,
        channels=channels,
        frame_count=frame_count,
        duration_seconds=duration,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        max_abs_dc_offset=dc,
        clipped_fraction=clipped,
        zero_crossing_rate=zcr,
        spectral_flatness=flatness,
        low_band_fraction=low,
        mid_band_fraction=mid,
        high_band_fraction=high,
    )


def require_startup_pcm_quality(
    pcm: bytes | bytearray | memoryview | np.ndarray,
    *,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
    channels: int = EXPECTED_CHANNELS,
) -> PCMQualityReport:
    """Return the report, raising :class:`PCMQualityError` on hard failure."""

    report = analyze_startup_pcm(
        pcm,
        sample_rate=sample_rate,
        channels=channels,
    )
    if not report.passed:
        raise PCMQualityError(report)
    return report


__all__ = [
    "PCMQualityError",
    "PCMQualityReport",
    "analyze_startup_pcm",
    "require_startup_pcm_quality",
]
