"""Musical, timing, and token invariants for the shared composition planner."""

from dataclasses import replace
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import composition as C  # noqa: E402


def event_signature(event: C.PianoRollEvent):
    position = event.position
    return (
        event.conditioning_key,
        position.frame,
        position.step_position,
        position.absolute_step,
        position.bar,
        position.step_in_bar,
        position.bpm,
        event.mood,
        event.key,
        event.section,
        event.chord,
        event.melody_note,
        event.active_notes,
        event.onset_notes,
        event.released_notes,
    )


class CompositionClockTests(unittest.TestCase):
    def test_clock_tracks_frames_steps_beats_and_bars(self):
        clock = C.CompositionClock()
        positions = [clock.advance(60) for _ in range(101)]

        self.assertEqual(positions[0].frame, 0)
        self.assertEqual(positions[0].absolute_step, 0)
        self.assertEqual(positions[25].absolute_step, 4)
        self.assertEqual(positions[25].beat_in_bar, 1)
        self.assertEqual(positions[100].absolute_step, 16)
        self.assertEqual(positions[100].bar, 1)
        self.assertEqual(positions[100].step_in_bar, 0)

    def test_faster_tempo_advances_farther_without_resetting_phase(self):
        slow = C.CompositionClock()
        fast = C.CompositionClock()
        for _ in range(250):
            slow.advance(60)
            fast.advance(110)

        self.assertEqual(slow.frame_index, fast.frame_index)
        self.assertGreater(fast.step_position, slow.step_position)
        before = slow.step_position
        slow.advance(110)
        self.assertGreater(slow.step_position, before)


class CompositionPlannerTests(unittest.TestCase):
    def test_chunking_cannot_change_rng_music_or_clock_metadata(self):
        whole = C.CompositionPlanner("chunk-stable")
        split = C.CompositionPlanner("chunk-stable")
        whole_events = whole.plan_frames("neutral", 700)
        split_events = split.plan_frames("neutral", 117)
        split_events += split.plan_frames("neutral", 229)
        split_events += split.plan_frames("neutral", 354)

        self.assertEqual(
            [event_signature(event) for event in split_events],
            [event_signature(event) for event in whole_events],
        )

    def test_defaults_are_mood_sensitive_and_override_is_clamped(self):
        for mood, bpm in C.MOOD_BPMS.items():
            event = C.CompositionPlanner(f"tempo-{mood}").plan_frames(mood, 1)[0]
            self.assertEqual(event.position.bpm, bpm)

        planner = C.CompositionPlanner("tempo-override")
        self.assertEqual(planner.configure(bpm=20).bpm, C.BPM_MIN)
        self.assertEqual(planner.plan_frames(None, 1)[0].position.bpm, C.BPM_MIN)
        self.assertEqual(planner.configure(bpm=999).bpm, C.BPM_MAX)
        planner.plan_frames(None, 10)
        self.assertEqual(planner.active_settings.bpm, C.BPM_MAX)
        self.assertIsNone(planner.configure(bpm=None).bpm)

    def test_runtime_controls_update_without_clock_reset(self):
        planner = C.CompositionPlanner("live-controls")
        planner.plan_frames("neutral", 37)
        frame_before = planner.frame_index
        step_before = planner.clock.step_position
        settings = planner.configure(
            station="  Rainy Cafe  ",
            bpm=84,
            groove=2,
            intensity=-1,
            melody_enabled=False,
            drums_enabled=False,
        )

        self.assertEqual(planner.frame_index, frame_before)
        self.assertEqual(planner.clock.step_position, step_before)
        self.assertEqual(settings.station, "rainy cafe")
        self.assertEqual(settings.groove, 1)
        self.assertEqual(settings.intensity, 0)
        planner.plan_frames(None, 20)
        self.assertEqual(planner.frame_index, frame_before + 20)
        self.assertEqual(planner.active_settings.bpm, 84)

    def test_form_sections_have_stable_boundaries_and_repeat(self):
        planner = C.CompositionPlanner("form")
        expected = {
            0: ("intro", 0),
            3: ("intro", 3),
            4: ("theme_a", 0),
            11: ("theme_a", 7),
            12: ("theme_b", 0),
            19: ("theme_b", 7),
            20: ("breakdown", 0),
            23: ("breakdown", 3),
            24: ("reprise", 0),
            31: ("reprise", 7),
            32: ("intro", 0),
        }
        for bar, (name, local_bar) in expected.items():
            section = planner.section_at(bar)
            self.assertEqual((section.name, section.bar_in_section), (name, local_bar))
        self.assertEqual(planner.section_at(32).cycle, 1)

    def test_chords_are_diatonic_varied_and_intensity_controls_voicing(self):
        planner = C.CompositionPlanner("harmony")
        for mood, scale in C.MODE_SCALES.items():
            chords = [planner.chord_at(mood, bar) for bar in range(C.FORM_BARS)]
            self.assertGreaterEqual(len({chord.degree for chord in chords}), 4, mood)
            for chord in chords:
                self.assertIn(chord.degree, range(7))
                self.assertTrue(
                    all((note - planner.tonic) % 12 in scale for note in chord.notes),
                    (mood, chord),
                )

        self.assertEqual(len(planner.chord_at("neutral", 0, intensity=0.1).notes), 2)
        self.assertEqual(len(planner.chord_at("neutral", 0, intensity=0.5).notes), 3)
        self.assertEqual(len(planner.chord_at("neutral", 0, intensity=0.9).notes), 4)

    def test_chord_voicings_stay_clear_and_move_economically(self):
        planner = C.CompositionPlanner("voice-leading")
        chords = [planner.chord_at("neutral", bar) for bar in range(C.FORM_BARS)]

        self.assertTrue(all(50 <= note <= 72 for chord in chords for note in chord.notes))
        motions = [
            sum(abs(left - right) for left, right in zip(first.notes, second.notes))
            for first, second in zip(chords, chords[1:])
        ]
        self.assertLessEqual(max(motions), 12)

    def test_harmony_is_articulated_with_real_air_between_comping_hits(self):
        planner = C.CompositionPlanner("articulated-harmony")
        events = planner.plan_frames("neutral", 110)
        settings = planner.active_settings
        harmony = [
            planner._harmony_at(settings, event.section, event.position, event.chord)
            for event in events
        ]

        self.assertTrue(any(notes for notes in harmony))
        self.assertTrue(any(not notes for notes in harmony))
        self.assertTrue(any(len(notes) == 1 for notes in harmony), "bass never sounded alone")

    def test_reprise_returns_to_a_progression_and_motif(self):
        planner = C.CompositionPlanner("actual-reprise")
        for mood in C.MODE_SCALES:
            theme = [planner.chord_at(mood, bar).degree for bar in range(4, 12)]
            reprise = [planner.chord_at(mood, bar).degree for bar in range(24, 32)]
            self.assertEqual(reprise, theme)
            for theme_phrase, reprise_phrase in zip(range(2, 6), range(12, 16)):
                self.assertEqual(
                    planner.phrase_degrees(mood, reprise_phrase),
                    planner.phrase_degrees(mood, theme_phrase),
                )

    def test_motifs_develop_but_each_phrase_keeps_a_tonic_cadence(self):
        for mood in C.MODE_SCALES:
            planner = C.CompositionPlanner(f"motif-{mood}")
            phrases = [planner.phrase_degrees(mood, index) for index in range(8)]
            self.assertGreater(len(set(phrases)), 1, mood)
            for phrase in phrases:
                sounding = [degree for degree in phrase if degree is not None]
                self.assertGreaterEqual(len(set(sounding)), 3, mood)
                self.assertTrue(all(0 <= degree <= 7 for degree in sounding), mood)
                self.assertEqual(sounding[-1], 0, mood)

    def test_harmony_melody_and_metadata_share_one_key_and_clock(self):
        planner = C.CompositionPlanner("one-clock")
        events = planner.plan_frames("somber", 900)
        scale = C.MODE_SCALES["somber"]

        self.assertTrue(any(event.position.bar > 0 for event in events))
        for event in events:
            self.assertEqual(event.section, planner.section_at(event.position.bar))
            self.assertTrue(
                all((note - planner.tonic) % 12 in scale for note in event.chord.notes)
            )
            if event.melody_note is not None:
                self.assertIn((event.melody_note - planner.tonic) % 12, scale)

    def test_structural_melody_onsets_are_current_chord_tones(self):
        for mood, scale in C.MODE_SCALES.items():
            planner = C.CompositionPlanner(f"chord-aware-{mood}")
            events = planner.plan_frames(mood, 1400)
            structural = [
                event
                for event in events
                if event.melody_note in event.onset_notes
                and event.position.phase_in_step < 0.3
                and event.position.step_in_bar % 4 == 0
            ]
            self.assertTrue(structural, mood)
            for event in structural:
                chord_classes = {
                    C._scale_pitch(planner.tonic, scale, event.chord.degree + offset) % 12
                    for offset in (0, 2, 4)
                }
                self.assertIn(event.melody_note % 12, chord_classes, (mood, event))

    def test_groove_changes_performance_timing_not_motif_content(self):
        straight = C.CompositionPlanner(
            "same-motif", settings=C.PlannerSettings(groove=0.45)
        )
        swung = C.CompositionPlanner(
            "same-motif", settings=C.PlannerSettings(groove=0.9)
        )
        for phrase in range(8):
            self.assertEqual(
                straight.phrase_degrees("neutral", phrase),
                swung.phrase_degrees("neutral", phrase),
            )
        self.assertGreater(swung._groove_delay(0.9), straight._groove_delay(0.45))

    def test_note_tokens_encode_onsets_sustains_releases_and_masks_exactly(self):
        planner = C.CompositionPlanner("token-states")
        events = planner.plan_frames("neutral", 180)

        self.assertTrue(events[0].onset_notes)
        self.assertTrue(any(event.released_notes for event in events))
        self.assertTrue(any(event.onset_notes for event in events[1:]))
        for event in events:
            self.assertNotIn(3, event.tokens)
            onsets = set(event.onset_notes)
            releases = set(event.released_notes)
            active = set(event.active_notes)
            for pitch, token in enumerate(event.tokens):
                if pitch in onsets:
                    self.assertEqual(token, 2)
                elif pitch in releases:
                    self.assertEqual(token, 0)
                elif pitch in active:
                    self.assertEqual(token, 1)
                else:
                    self.assertEqual(token, -1)

    def test_disabling_or_unconstraining_releases_then_masks_the_guide(self):
        for update in (
            {"melody_enabled": False},
            {"guide_mode": C.GuideMode.UNCONSTRAINED},
            {"guide_mode": "off"},
        ):
            planner = C.CompositionPlanner(f"off-{update}")
            planner.plan_frames("neutral", 3)
            planner.configure(**update)
            events = planner.plan_frames(None, 30)
            release_index = next(
                index for index, event in enumerate(events) if event.released_notes
            )
            release = events[release_index]
            self.assertTrue(all(release.tokens[note] == 0 for note in release.released_notes))
            self.assertTrue(events[release_index + 1].unconstrained)
            self.assertEqual(events[release_index + 1].active_notes, ())

    def test_reenabling_guide_starts_fresh_onsets(self):
        planner = C.CompositionPlanner(
            "reenable", settings=C.PlannerSettings(melody_enabled=False)
        )
        self.assertTrue(planner.plan_frames("neutral", 1)[0].unconstrained)
        planner.configure(melody_enabled=True)
        events = planner.plan_frames(None, 20)
        first_guided = next(event for event in events if event.active_notes)
        self.assertEqual(set(first_guided.onset_notes), set(first_guided.active_notes))

    def test_production_drum_control_is_unconditional_or_explicitly_off(self):
        enabled = C.CompositionPlanner("drums").plan_frames("lively", 150)
        self.assertTrue(all(event.drum == -1 for event in enabled))

        disabled = C.CompositionPlanner(
            "no-drums", settings=C.PlannerSettings(drums_enabled=False)
        ).plan_frames("neutral", 100)
        self.assertTrue(all(event.drum == 0 for event in disabled))

    def test_experimental_strict_drum_prior_fills_hits_and_non_hits(self):
        strict = C.CompositionPlanner(
            "strict-drums", settings=C.PlannerSettings(strict_drums=True)
        ).plan_frames("lively", 150)

        self.assertTrue(any(event.drum == 1 for event in strict))
        self.assertTrue(any(event.drum == 0 for event in strict))
        self.assertNotIn(-1, {event.drum for event in strict})
        self.assertTrue(
            all(event.position.phase_in_step < 0.3 for event in strict if event.drum == 1)
        )

    def test_run_length_encoding_preserves_every_conditioning_frame(self):
        planner = C.CompositionPlanner("runs")
        runs = planner.plan("neutral", 400)

        self.assertEqual(sum(frames for _event, frames in runs), 400)
        self.assertTrue(all(frames > 0 for _event, frames in runs))
        self.assertLess(len(runs), 400)
        self.assertTrue(all(a != b for (a, _), (b, _) in zip(runs, runs[1:])))

    def test_events_hash_by_conditioning_and_validate_shape_and_vocabulary(self):
        tokens = (-1,) * 128
        first = C.PianoRollEvent(tokens, drum=-1)
        second = replace(first, mood="lively", key="C major")
        changed_drum = replace(first, drum=1)
        from_mutable_input = C.PianoRollEvent(list(tokens), drum=-1)

        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertNotEqual(first, changed_drum)
        self.assertIsInstance(from_mutable_input.tokens, tuple)
        self.assertEqual(hash(first), hash(from_mutable_input))
        with self.assertRaises(ValueError):
            C.PianoRollEvent((-1,) * 127)
        with self.assertRaises(ValueError):
            C.PianoRollEvent((3,) + (-1,) * 127)
        with self.assertRaises(ValueError):
            C.PianoRollEvent(tokens, drum=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
