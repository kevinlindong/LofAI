"""Standalone deterministic checks for the startup PCM quality gate."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_quality import (  # noqa: E402
    PCMQualityError,
    analyze_startup_pcm,
    require_startup_pcm_quality,
)


SAMPLE_RATE = 48_000
CHANNELS = 2
FRAMES = SAMPLE_RATE


def _pcm_bytes(stereo: np.ndarray) -> bytes:
    samples = np.clip(stereo, -1.0, 1.0)
    return np.ascontiguousarray(samples * 32_767.0, dtype=np.int16).tobytes()


def melodic_pcm() -> bytes:
    """A one-second, modest-level lo-fi chord/bass/melody phrase."""

    time = np.arange(FRAMES, dtype=np.float64) / SAMPLE_RATE
    fade_samples = int(0.02 * SAMPLE_RATE)
    envelope = np.ones(FRAMES, dtype=np.float64)
    envelope[:fade_samples] = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
    envelope[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples, endpoint=False)

    bass = 0.17 * np.sin(2.0 * np.pi * 110.0 * time)
    chord = sum(
        0.055 * np.sin(2.0 * np.pi * frequency * time + phase)
        for frequency, phase in (
            (220.0, 0.1),
            (261.63, 0.7),
            (329.63, 1.2),
            (440.0, 0.4),
        )
    )
    # Four connected notes make the fixture melodic rather than merely tonal.
    melody = np.zeros(FRAMES, dtype=np.float64)
    note_length = FRAMES // 4
    for index, frequency in enumerate((440.0, 523.25, 493.88, 392.0)):
        start = index * note_length
        stop = FRAMES if index == 3 else (index + 1) * note_length
        local_time = time[start:stop] - time[start]
        local_envelope = np.sin(np.linspace(0.0, np.pi, stop - start)) ** 0.35
        melody[start:stop] = (
            0.075
            * local_envelope
            * np.sin(2.0 * np.pi * frequency * local_time)
        )

    left = envelope * (bass + chord + melody)
    right = envelope * (
        0.96 * bass
        + sum(
            0.052 * np.sin(2.0 * np.pi * frequency * time + phase + 0.18)
            for frequency, phase in (
                (220.0, 0.1),
                (261.63, 0.7),
                (329.63, 1.2),
                (440.0, 0.4),
            )
        )
        + 0.92 * melody
    )
    return _pcm_bytes(np.column_stack((left, right)))


class AudioQualityTests(unittest.TestCase):
    def test_clean_melodic_pcm_passes(self):
        report = analyze_startup_pcm(melodic_pcm())

        self.assertTrue(report.passed, report)
        self.assertEqual(report.issues, ())
        self.assertLess(report.zero_crossing_rate, 0.10)
        self.assertLess(report.spectral_flatness, 0.10)

    def test_silence_fails_as_silence(self):
        report = analyze_startup_pcm(np.zeros((FRAMES, CHANNELS), dtype=np.int16))

        self.assertFalse(report.passed)
        self.assertIn("silence", report.issues)

    def test_constant_dc_fails_for_dc_offset(self):
        dc = np.full((FRAMES, CHANNELS), int(0.25 * 32_767), dtype=np.int16)
        report = analyze_startup_pcm(dc)

        self.assertFalse(report.passed)
        self.assertIn("dc_offset", report.issues)

    def test_rail_clipped_pcm_fails_for_clipping(self):
        clipped = np.empty((FRAMES, CHANNELS), dtype=np.int16)
        clipped[::2] = 32_767
        clipped[1::2] = -32_768
        report = analyze_startup_pcm(clipped)

        self.assertFalse(report.passed)
        self.assertIn("clipping", report.issues)
        self.assertIn("rms_too_high", report.issues)

    def test_seeded_white_noise_fails_for_noise_signature(self):
        rng = np.random.default_rng(20260830)
        noise = rng.normal(0.0, 0.14, size=(FRAMES, CHANNELS))
        report = analyze_startup_pcm(_pcm_bytes(noise))

        self.assertFalse(report.passed)
        self.assertIn("white_noise", report.issues)
        self.assertGreater(report.zero_crossing_rate, 0.45)
        self.assertGreater(report.spectral_flatness, 0.90)
        self.assertGreater(report.high_band_fraction, 0.70)

    def test_malformed_and_short_pcm_fail_structural_checks(self):
        malformed = analyze_startup_pcm(b"\x00")
        short = analyze_startup_pcm(np.zeros((100, CHANNELS), dtype=np.int16))
        invalid_rate = analyze_startup_pcm(
            np.zeros((FRAMES, CHANNELS), dtype=np.int16), sample_rate=0
        )

        self.assertIn("invalid_byte_length", malformed.issues)
        self.assertIn("too_short", short.issues)
        self.assertIn("invalid_sample_rate", invalid_rate.issues)

    def test_raise_on_failure_helper_returns_good_report(self):
        report = require_startup_pcm_quality(melodic_pcm())
        self.assertTrue(report.passed)

        with self.assertRaises(PCMQualityError) as caught:
            require_startup_pcm_quality(np.zeros((FRAMES, CHANNELS), dtype=np.int16))
        self.assertIn("silence", caught.exception.report.issues)


if __name__ == "__main__":
    unittest.main(verbosity=2)
