"""Detecting and repairing a take whose noise floor drifts upward.

MRT2 continues its own recurrent audio: whatever it just played is the
context for what it plays next. On sparse stations that feedback has a
failure attractor - once a bright sustained texture (tape hiss, cymbal wash)
enters the roughly 20-second context window, the model tends to continue and
amplify it. Measured on a 12-minute rainy-piano take, the level of the
quietest 50 ms blocks rose from -42.7 to -32.6 dBFS while their 6-12 kHz
energy share grew from 3% to 19%: a hiss bed that was not there at the
start. Busier stations (dusty-beats) held a stable floor for the same
duration, so this is a per-take failure, not a constant.

``TakeFloorMonitor`` watches the stream the listener actually receives. It
profiles the quietest blocks - the gaps between notes, where a hiss bed is
exposed - and compares a frozen early-take baseline against a trailing
window. Both the overall gap floor and its high-band level must rise
together, sustained for several seconds, before it reports drift: musical
passages getting denser moves the gap floor but not, in this genre, its
high band. The repair is a server-side equal-power crossfade onto a fresh
recurrent state (same station, new seed), which reads as a radio track
change rather than a dropout: PCM flow, session identity, and the client
transport are untouched.
"""

from __future__ import annotations

from collections import deque

import numpy as np

SAMPLE_RATE = 48_000
CHANNELS = 2

BLOCK_SECONDS = 0.05
BLOCK_FRAMES = int(SAMPLE_RATE * BLOCK_SECONDS)

# The floor is the quietest tenth of blocks in a window: between-note gaps,
# not the performance.
FLOOR_PERCENTILE = 10.0

# Skip the first seconds of a take (transport start, style landing), then
# freeze a baseline over the following stretch.
BASELINE_SKIP_SECONDS = 10.0
BASELINE_SECONDS = 60.0
TRAILING_SECONDS = 60.0

# A hiss bed is, concretely, high-frequency noise in the quietest blocks
# that was not there at the start of the take. The high band of the floor
# must rise this far over its own baseline, and be absolutely loud enough
# to hear, before a chunk votes for drift. The overall floor level is
# reported for observability but deliberately not required: a captured live
# failure grew +22 dB of high-band hiss while its overall floor rose only
# +1 dB (the bed brightened long before it lifted), and healthy takes'
# high-band floors wander far less. Measured runaways rose +17 to +28 dB in
# band and stayed; the strongest healthy excursion (brushed-cymbal wash)
# briefly reached +17 dB and receded within a window.
HIGH_BAND_RISE_DB = 10.0
MIN_AUDIBLE_HIGH_DBFS = -52.0
# Drift must hold in most chunk evaluations across this window. A fraction
# over a window, not a consecutive streak: one quiet evaluation must not
# reset the clock on a take that has been hissing for a minute.
SUSTAIN_WINDOW_CHUNKS = 100
SUSTAIN_FRACTION = 0.8

HIGH_BAND_LOW_HZ = 5_000.0
HIGH_BAND_HIGH_HZ = 14_000.0

_EPS = 1e-12


def _dbfs(value: float) -> float:
    return 20.0 * float(np.log10(value + _EPS))


def _floor_profile(rms_values, high_values) -> tuple[float, float]:
    """Level and high-band level of the quietest blocks.

    Blocks are selected by overall RMS - the between-note gaps - and the
    high-band level is measured on those same blocks. Selecting the high
    band independently would find the darkest blocks (a pure bass note)
    rather than the spectrum of the floor itself.
    """
    rms = np.asarray(rms_values, dtype=np.float64)
    high = np.asarray(high_values, dtype=np.float64)
    take = max(1, int(rms.size * FLOOR_PERCENTILE / 100.0))
    quiet = np.argsort(rms)[:take]
    return float(np.median(rms[quiet])), float(np.median(high[quiet]))


class TakeFloorMonitor:
    """Tracks the quiet-block floor of one take from its int16 PCM."""

    def __init__(self):
        blocks = int(TRAILING_SECONDS / BLOCK_SECONDS)
        self._trailing_rms: deque[float] = deque(maxlen=blocks)
        self._trailing_high: deque[float] = deque(maxlen=blocks)
        self._baseline_rms: list[float] = []
        self._baseline_high: list[float] = []
        self._baseline_floor: float | None = None
        self._baseline_high_floor: float | None = None
        self._seen_blocks = 0
        self._drift_votes: deque[bool] = deque(maxlen=SUSTAIN_WINDOW_CHUNKS)
        self._remainder = np.empty(0, dtype=np.float32)
        window = np.hanning(BLOCK_FRAMES).astype(np.float32)
        self._window = window
        freqs = np.fft.rfftfreq(BLOCK_FRAMES, 1.0 / SAMPLE_RATE)
        self._high_bins = (freqs >= HIGH_BAND_LOW_HZ) & (freqs <= HIGH_BAND_HIGH_HZ)
        # Parseval-style scale so band energy reads in sample units.
        self._fft_scale = 1.0 / (np.sum(window**2) * BLOCK_FRAMES / 2.0)

    def reset(self):
        self.__init__()

    def observe(self, pcm: bytes):
        """Feed one interleaved int16 stereo chunk in play order."""
        frame_bytes = 2 * CHANNELS
        usable_bytes = len(pcm) - (len(pcm) % frame_bytes)
        if usable_bytes <= 0:
            self._evaluate()
            return
        samples = np.frombuffer(pcm[:usable_bytes], dtype=np.int16)
        mono = (
            samples.reshape(-1, CHANNELS).mean(axis=1, dtype=np.float32) / 32768.0
        )
        if self._remainder.size:
            mono = np.concatenate([self._remainder, mono])
        usable = mono.size // BLOCK_FRAMES * BLOCK_FRAMES
        self._remainder = mono[usable:]
        if usable:
            blocks = mono[:usable].reshape(-1, BLOCK_FRAMES)
            rms = np.sqrt(np.mean(blocks**2, axis=1) + _EPS)
            spectra = np.abs(np.fft.rfft(blocks * self._window, axis=1)) ** 2
            high = np.sqrt(
                np.sum(spectra[:, self._high_bins], axis=1) * self._fft_scale + _EPS
            )
            for block_rms, block_high in zip(rms, high):
                self._observe_block(float(block_rms), float(block_high))
        self._evaluate()

    def _observe_block(self, rms: float, high: float):
        self._seen_blocks += 1
        seconds = self._seen_blocks * BLOCK_SECONDS
        if seconds <= BASELINE_SKIP_SECONDS:
            return
        if seconds <= BASELINE_SKIP_SECONDS + BASELINE_SECONDS:
            self._baseline_rms.append(rms)
            self._baseline_high.append(high)
            return
        if self._baseline_floor is None:
            self._baseline_floor, self._baseline_high_floor = _floor_profile(
                self._baseline_rms, self._baseline_high
            )
            self._baseline_rms = []
            self._baseline_high = []
        self._trailing_rms.append(rms)
        self._trailing_high.append(high)

    def _evaluate(self):
        if (
            self._baseline_floor is None
            or len(self._trailing_rms) < self._trailing_rms.maxlen
        ):
            return
        floor, high_floor = _floor_profile(self._trailing_rms, self._trailing_high)
        del floor  # reported by describe(); the decision is spectral
        drifted = (
            _dbfs(high_floor) >= MIN_AUDIBLE_HIGH_DBFS
            and _dbfs(high_floor) - _dbfs(self._baseline_high_floor)
            >= HIGH_BAND_RISE_DB
        )
        self._drift_votes.append(drifted)

    @property
    def drifted(self) -> bool:
        votes = self._drift_votes
        if len(votes) < votes.maxlen:
            return False
        return sum(votes) >= SUSTAIN_FRACTION * votes.maxlen

    def describe(self) -> str:
        if self._baseline_floor is None or not self._trailing_rms:
            return "warming"
        floor, high_floor = _floor_profile(self._trailing_rms, self._trailing_high)
        return (
            f"floor {_dbfs(floor):.1f} dBFS, high band {_dbfs(high_floor):.1f} dBFS "
            f"(baseline {_dbfs(self._baseline_floor):.1f} / "
            f"{_dbfs(self._baseline_high_floor):.1f} dBFS)"
        )


def crossfade_pcm(old_pcm: bytes, new_pcm: bytes) -> bytes:
    """Equal-power crossfade between two equal-length int16 stereo chunks."""
    old = np.frombuffer(old_pcm, dtype=np.int16).astype(np.float32)
    new = np.frombuffer(new_pcm, dtype=np.int16).astype(np.float32)
    frames = min(old.size, new.size) // CHANNELS
    old = old[: frames * CHANNELS].reshape(frames, CHANNELS)
    new = new[: frames * CHANNELS].reshape(frames, CHANNELS)
    theta = np.linspace(0.0, np.pi / 2.0, frames, dtype=np.float32)[:, None]
    mixed = old * np.cos(theta) + new * np.sin(theta)
    return (
        np.clip(np.round(mixed), -32768, 32767)
        .astype(np.int16)
        .reshape(-1)
        .tobytes()
    )
