"""Listener-control and curated-style behavior without loading MRT2."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from music_controls import MusicControls  # noqa: E402
import styles  # noqa: E402


class MusicControlTests(unittest.TestCase):
    def test_legacy_client_keeps_custom_mood_and_instrument(self):
        controls = MusicControls.initial("somber", "piano")
        self.assertEqual(controls.station, "custom")
        self.assertEqual((controls.mood, controls.instrument), ("somber", "piano"))

    def test_station_supplies_one_coherent_set_of_defaults(self):
        controls = MusicControls.initial(station="sunlit-groove")
        preset = styles.STATIONS["sunlit-groove"]
        self.assertEqual(controls.station, preset.slug)
        self.assertEqual(controls.bpm, preset.bpm)
        self.assertEqual(controls.prompt(), preset.prompt)

    def test_controls_are_clamped_and_invalid_types_do_not_toggle(self):
        controls = MusicControls.initial(station="dusty-beats").update(
            {
                "bpm": 999,
                "groove": -4,
                "intensity": "0.75",
                "melody": "false",
                "drums": False,
            }
        )
        self.assertEqual(controls.bpm, 110)
        self.assertEqual(controls.groove, 0.0)
        self.assertEqual(controls.intensity, 0.75)
        self.assertTrue(controls.melody)
        self.assertFalse(controls.drums)

    def test_legacy_axis_change_becomes_a_custom_style(self):
        controls = MusicControls.initial(station="rainy-piano")
        controls = controls.update({"instrument": "brass"})
        self.assertEqual(controls.station, "custom")
        self.assertIn("muted trumpet", controls.prompt())

    def test_named_station_wins_over_stale_legacy_style_fields(self):
        controls = MusicControls.initial(
            station="rainy-piano",
            payload={
                "station": "rainy-piano",
                "mood": "lively",
                "instrument": "brass",
            },
        )

        preset = styles.STATIONS["rainy-piano"]
        self.assertEqual(controls.station, preset.slug)
        self.assertEqual(controls.mood, preset.mood)
        self.assertEqual(controls.instrument, preset.instrument)
        self.assertEqual(controls.prompt(), preset.prompt)

    def test_audio_reference_requires_an_existing_station_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dusty-beats.wav")
            with open(path, "wb") as output:
                output.write(b"RIFF")
            with patch.dict(os.environ, {"MRT_STYLE_REFERENCE_DIR": directory}):
                self.assertEqual(
                    styles.audio_reference_for("dusty-beats"), os.path.realpath(path)
                )
                self.assertIsNone(styles.audio_reference_for("jazz-cafe"))
                self.assertIsNone(styles.audio_reference_for("custom"))

    def test_public_options_match_protocol_ranges(self):
        options = styles.public_options()
        self.assertEqual(options["limits"]["bpm"], [60, 110])
        self.assertIn(options["defaultStation"], styles.STATIONS)
        self.assertEqual(
            {station["slug"] for station in options["stations"]},
            set(styles.STATIONS),
        )
        self.assertEqual(options["limits"]["adherence"], [0.0, 1.0])
        self.assertEqual(options["limits"]["variation"], [0.0, 1.0])
        self.assertEqual(options["customStation"], styles.CUSTOM_STATION)

    def test_free_text_prompt_is_wrapped_in_the_lofi_scaffold(self):
        controls = MusicControls.initial().update({"customPrompt": "rainy tokyo night"})
        self.assertEqual(controls.station, "custom")
        prompt = controls.prompt()
        self.assertTrue(prompt.startswith("instrumental lo-fi,"))
        self.assertIn("rainy tokyo night", prompt)

    def test_free_text_prompt_selects_custom_station(self):
        controls = MusicControls.initial(station="dusty-beats")
        # A typed prompt with no explicit station switches to the custom mix.
        controls = controls.update({"customPrompt": "warm vinyl saxophone"})
        self.assertEqual(controls.station, "custom")

    def test_scaffold_does_not_double_lofi(self):
        self.assertEqual(
            styles.scaffold_custom_prompt("lofi beats to relax to"),
            "lofi beats to relax to",
        )
        self.assertIsNone(styles.scaffold_custom_prompt("   "))
        self.assertIsNone(styles.scaffold_custom_prompt(None))

    def test_free_text_prompt_is_length_capped(self):
        long_text = "piano " * 60
        controls = MusicControls.initial().update({"customPrompt": long_text})
        self.assertLessEqual(
            len(controls.customPrompt), styles.MAX_CUSTOM_PROMPT_CHARS
        )

    def test_named_station_clears_stale_custom_prompt(self):
        controls = MusicControls.initial().update({"customPrompt": "spacey pads"})
        self.assertEqual(controls.station, "custom")
        controls = controls.update({"station": "jazz-cafe"})
        self.assertEqual(controls.station, "jazz-cafe")
        self.assertEqual(controls.customPrompt, "")
        self.assertEqual(controls.prompt(), styles.STATIONS["jazz-cafe"].prompt)

    def test_granular_dials_clamp_and_default_to_neutral(self):
        controls = MusicControls.initial()
        self.assertEqual(controls.adherence, 0.5)
        self.assertEqual(controls.variation, 0.5)
        controls = controls.update({"adherence": 9, "variation": -3})
        self.assertEqual(controls.adherence, 1.0)
        self.assertEqual(controls.variation, 0.0)

    def test_sampling_overrides_are_monotonic_and_centered(self):
        neutral = MusicControls.initial().sampling_overrides()
        self.assertAlmostEqual(neutral["cfg_musiccoca_scale"], 1.075, places=3)
        low = MusicControls.initial().update({"adherence": 0.0}).sampling_overrides()
        high = MusicControls.initial().update({"adherence": 1.0}).sampling_overrides()
        self.assertLess(low["cfg_musiccoca_scale"], high["cfg_musiccoca_scale"])
        calm = MusicControls.initial().update({"variation": 0.0}).sampling_overrides()
        wild = MusicControls.initial().update({"variation": 1.0}).sampling_overrides()
        self.assertLess(calm["temperature_scale"], wild["temperature_scale"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
