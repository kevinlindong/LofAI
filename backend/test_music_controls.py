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

    def test_named_station_cannot_keep_a_contradictory_style_identity(self):
        controls = MusicControls.initial(
            station="rainy-piano",
            payload={
                "station": "rainy-piano",
                "mood": "lively",
                "instrument": "brass",
            },
        )

        self.assertEqual(controls.station, styles.CUSTOM_STATION)
        self.assertEqual(controls.mood, "lively")
        self.assertEqual(controls.instrument, "brass")
        self.assertIn("muted trumpet", controls.prompt())

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
