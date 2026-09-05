from pathlib import Path
import sys
import unittest

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.audio_metrics import METRICS_DISCLAIMER, analyze_audio  # noqa: E402


SAMPLE_RATE = 48_000


class AudioMetricsTests(unittest.TestCase):
    def test_sine_has_expected_level_and_bandwidth(self):
        time = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
        mono = 0.25 * np.sin(2 * np.pi * 440.0 * time)
        stereo = np.column_stack((mono, mono))

        metrics = analyze_audio(stereo, SAMPLE_RATE)

        self.assertAlmostEqual(metrics.peak_dbfs, -12.04, delta=0.1)
        self.assertAlmostEqual(metrics.rms_dbfs, -15.05, delta=0.1)
        self.assertLess(metrics.spectral_flatness, 0.01)
        self.assertLess(metrics.high_band_energy_fraction, 0.001)
        self.assertEqual(metrics.clipped_sample_fraction, 0.0)
        self.assertIn("not a music-quality", METRICS_DISCLAIMER)

    def test_noise_is_flatter_and_brighter_than_tone(self):
        rng = np.random.default_rng(123)
        noise = rng.normal(0.0, 0.1, (SAMPLE_RATE, 2))
        time = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
        tone = np.column_stack([0.1 * np.sin(2 * np.pi * 220 * time)] * 2)

        noise_metrics = analyze_audio(noise, SAMPLE_RATE)
        tone_metrics = analyze_audio(tone, SAMPLE_RATE)

        self.assertGreater(noise_metrics.spectral_flatness, 0.8)
        self.assertGreater(
            noise_metrics.high_band_energy_fraction,
            tone_metrics.high_band_energy_fraction,
        )

    def test_click_track_has_onset_activity_and_loop_recurrence(self):
        seconds = 4
        mono = np.zeros(SAMPLE_RATE * seconds, dtype=np.float64)
        for start in range(0, mono.size, SAMPLE_RATE // 2):
            mono[start : start + 64] = np.hanning(64) * 0.5
        stereo = np.column_stack((mono, mono))

        metrics = analyze_audio(stereo, SAMPLE_RATE)

        self.assertGreater(metrics.onset_proxy_per_second, 0.5)
        self.assertGreater(metrics.repetition_peak_correlation, 0.9)
        self.assertAlmostEqual(metrics.repetition_peak_lag_seconds, 0.5, delta=0.1)

    def test_int16_clipping_and_silence_are_safe(self):
        clipped = np.tile(np.array([[-32768, 32767]], dtype=np.int16), (1000, 1))
        metrics = analyze_audio(clipped, SAMPLE_RATE)
        self.assertGreater(metrics.clipped_sample_fraction, 0.99)

        silence = analyze_audio(np.zeros((100, 2), dtype=np.int16), SAMPLE_RATE)
        self.assertEqual(silence.spectral_flatness, 0.0)
        self.assertEqual(silence.repetition_peak_correlation, 0.0)
        self.assertEqual(silence.peak_dbfs, float("-inf"))

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            analyze_audio(np.array([], dtype=np.float32), SAMPLE_RATE)
        with self.assertRaises(ValueError):
            analyze_audio(np.zeros(10), 0)
        with self.assertRaises(ValueError):
            analyze_audio(np.array([0.0, np.nan]), SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
