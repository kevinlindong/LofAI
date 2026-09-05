"""Deterministic, frame-accurate musical planning for Magenta RealTime 2.

MRT2 is very good at performing a conditioning signal, but it is not a
long-range song planner.  This module supplies the missing shared musical
clock.  Harmony, melody, sections, and optional drum intent are all derived
from the same bar/step position, so they cannot quietly drift apart.

The public output is :class:`PianoRollEvent`.  Its 128 note tokens use MRT2's
native vocabulary exactly:

* ``-1`` - unspecified; the model may decide what happens at that pitch
* ``0``  - explicitly off
* ``1``  - held from the preceding frame
* ``2``  - onset on this frame

Events are immutable and hashable by conditioning content.  Descriptive
metadata is deliberately excluded from equality so repeated conditioning can
be cached and run-length encoded efficiently by the inference layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import itertools
import math
import random
from typing import Final, Iterable


FRAME_RATE: Final = 25
BEATS_PER_BAR: Final = 4
STEPS_PER_BEAT: Final = 4
STEPS_PER_BAR: Final = BEATS_PER_BAR * STEPS_PER_BEAT
PHRASE_BARS: Final = 2
PHRASE_STEPS: Final = PHRASE_BARS * STEPS_PER_BAR

BPM_MIN: Final = 60.0
BPM_MAX: Final = 110.0
MOOD_BPMS: Final = {
    "somber": 70.0,
    "neutral": 78.0,
    "lively": 92.0,
}

# Seven pitch classes are used for harmony.  The octave is appended for the
# compatibility melody API and for convenient degree calculations.
MODE_SCALES: Final = {
    "somber": (0, 2, 3, 5, 7, 8, 10),  # natural minor
    "neutral": (0, 2, 4, 5, 7, 9, 11),  # major
    "lively": (0, 2, 4, 5, 7, 9, 11),  # major
}
SCALES: Final = {mood: (*scale, 12) for mood, scale in MODE_SCALES.items()}

# Roots sit in the bass/harmony register while the lead begins two octaves
# above.  A session keeps its root when mood changes; major/minor colour can
# change without an unrelated key jump.
KEY_ROOTS: Final = (33, 36, 38, 41, 43)  # A1, C2, D2, F2, G2
PITCH_NAMES: Final = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


class GuideMode(str, Enum):
    """How the note-conditioning channel should be used."""

    GUIDED = "guided"
    UNCONSTRAINED = "unconstrained"
    OFF = "off"


@dataclass(frozen=True)
class PlannerSettings:
    """Runtime controls.  Updating them never resets the composition clock."""

    station: str = "lofi"
    mood: str = "neutral"
    bpm: float | None = None
    groove: float = 0.55
    intensity: float = 0.5
    melody_enabled: bool = True
    drums_enabled: bool = True
    # MRT2's published production use of the drum channel is on/off: masked
    # means drum-unconditional, zero means drumless. A strict binary pulse is
    # retained only as an explicit listening-test experiment because the model
    # authors report direct hit control is impractical at end-to-end latency.
    strict_drums: bool = False
    guide_mode: GuideMode = GuideMode.GUIDED

    @property
    def resolved_bpm(self) -> float:
        return self.bpm if self.bpm is not None else MOOD_BPMS[self.mood]


@dataclass(frozen=True)
class ClockPosition:
    frame: int
    step_position: float
    absolute_step: int
    bar: int
    step_in_bar: int
    beat_in_bar: int
    phase_in_step: float
    bpm: float


@dataclass(frozen=True)
class Section:
    name: str
    cycle: int
    bar_in_form: int
    bar_in_section: int
    length_bars: int


@dataclass(frozen=True)
class Chord:
    degree: int
    symbol: str
    notes: tuple[int, ...]


@dataclass(frozen=True)
class MotifNote:
    start: int
    duration: int
    degree: int


@dataclass(frozen=True)
class PianoRollEvent:
    """One frame's hashable MRT conditioning plus optional musical metadata."""

    tokens: tuple[int, ...]
    drum: int = -1
    position: ClockPosition | None = field(default=None, compare=False, hash=False)
    mood: str | None = field(default=None, compare=False, hash=False)
    key: str | None = field(default=None, compare=False, hash=False)
    section: Section | None = field(default=None, compare=False, hash=False)
    chord: Chord | None = field(default=None, compare=False, hash=False)
    melody_note: int | None = field(default=None, compare=False, hash=False)
    active_notes: tuple[int, ...] = field(default=(), compare=False, hash=False)
    onset_notes: tuple[int, ...] = field(default=(), compare=False, hash=False)
    released_notes: tuple[int, ...] = field(default=(), compare=False, hash=False)

    def __post_init__(self):
        # Type hints do not stop a caller from passing a mutable list.  Freeze
        # it here so the public ``tokens`` contract and event hashability hold
        # at runtime as well as under static type checking.
        tokens = tuple(self.tokens)
        object.__setattr__(self, "tokens", tokens)
        if len(tokens) != 128:
            raise ValueError(f"MRT piano roll must have 128 tokens, got {len(tokens)}")
        if any(token not in (-1, 0, 1, 2) for token in tokens):
            raise ValueError("MRT piano-roll tokens must be one of -1, 0, 1, or 2")
        if self.drum not in (-1, 0, 1):
            raise ValueError("MRT drum intent must be -1, 0, or 1")

    @property
    def unconstrained(self) -> bool:
        return all(token == -1 for token in self.tokens)

    @property
    def conditioning_key(self) -> tuple[tuple[int, ...], int]:
        return self.tokens, self.drum


PianoRollRun = tuple[PianoRollEvent, int]


@dataclass(frozen=True)
class _SectionSpec:
    name: str
    bars: int
    density: float
    register: int = 0


FORM: Final = (
    _SectionSpec("intro", 4, 0.55, 0),
    _SectionSpec("theme_a", 8, 0.92, 0),
    _SectionSpec("theme_b", 8, 1.08, 0),
    _SectionSpec("breakdown", 4, 0.48, 0),
    _SectionSpec("reprise", 8, 0.88, 0),
)
FORM_BARS: Final = sum(section.bars for section in FORM)

# Each tuple is a diatonic chord degree per bar.  Variants are selected by a
# stable station/session/form-cycle seed rather than by mutable global RNG.
PROGRESSIONS: Final = {
    "somber": (
        (0, 5, 2, 6, 0, 3, 5, 4),  # i - VI - III - VII / i - iv - VI - v
        (0, 3, 6, 2, 5, 3, 4, 0),
        (0, 6, 5, 3, 0, 2, 4, 0),
    ),
    "neutral": (
        (0, 5, 3, 4, 0, 2, 3, 4),  # I - vi - IV - V
        (0, 2, 5, 3, 1, 4, 0, 4),
        (0, 3, 1, 4, 5, 3, 4, 0),
    ),
    "lively": (
        (0, 4, 5, 3, 0, 3, 4, 4),  # I - V - vi - IV
        (0, 3, 1, 4, 0, 5, 3, 4),
        (5, 3, 0, 4, 1, 4, 0, 4),
    ),
}

ROMAN: Final = {
    "somber": ("i", "ii°", "III", "iv", "v", "VI", "VII"),
    "neutral": ("I", "ii", "iii", "IV", "V", "vi", "vii°"),
    "lively": ("I", "ii", "iii", "IV", "V", "vi", "vii°"),
}

# Two-bar motifs described in sixteenth-note steps.  They intentionally leave
# breathing room; section, phrase, groove, and intensity transformations below
# create development without destroying the recognizable contour.
BASE_MOTIFS: Final = {
    "somber": (
        MotifNote(0, 3, 0),
        MotifNote(6, 2, 2),
        MotifNote(10, 3, 3),
        MotifNote(16, 4, 4),
        MotifNote(23, 2, 2),
        MotifNote(28, 3, 0),
    ),
    "neutral": (
        MotifNote(0, 2, 0),
        MotifNote(4, 2, 1),
        MotifNote(7, 2, 2),
        MotifNote(11, 3, 4),
        MotifNote(16, 3, 3),
        MotifNote(21, 2, 4),
        MotifNote(24, 2, 2),
        MotifNote(29, 2, 0),
    ),
    "lively": (
        MotifNote(0, 2, 0),
        MotifNote(3, 2, 1),
        MotifNote(6, 2, 2),
        MotifNote(9, 2, 4),
        MotifNote(13, 2, 2),
        MotifNote(16, 2, 4),
        MotifNote(19, 2, 5),
        MotifNote(22, 2, 4),
        MotifNote(25, 2, 2),
        MotifNote(29, 2, 0),
    ),
}


def stable_seed(value: str) -> int:
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def normalize_mood(mood: str) -> str:
    return mood if mood in MODE_SCALES else "neutral"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _station_name(value: str) -> str:
    value = str(value).strip().lower()
    return value[:64] or "lofi"


def _scale_pitch(tonic: int, scale: tuple[int, ...], degree: int) -> int:
    octave, index = divmod(degree, 7)
    return tonic + octave * 12 + scale[index]


class CompositionClock:
    """A continuous fractional-step clock shared by all planned controls."""

    def __init__(self, frame_rate: int = FRAME_RATE):
        if frame_rate <= 0:
            raise ValueError("frame rate must be positive")
        self.frame_rate = frame_rate
        self.frame_index = 0
        self.step_position = 0.0
        self.last_position: ClockPosition | None = None

    def position(self, bpm: float) -> ClockPosition:
        bpm = _clamp(bpm, BPM_MIN, BPM_MAX)
        absolute_step = math.floor(self.step_position + 1e-10)
        step_in_bar = absolute_step % STEPS_PER_BAR
        return ClockPosition(
            frame=self.frame_index,
            step_position=self.step_position,
            absolute_step=absolute_step,
            bar=absolute_step // STEPS_PER_BAR,
            step_in_bar=step_in_bar,
            beat_in_bar=step_in_bar // STEPS_PER_BEAT,
            phase_in_step=self.step_position - absolute_step,
            bpm=bpm,
        )

    def advance(self, bpm: float) -> ClockPosition:
        position = self.position(bpm)
        self.last_position = position
        self.frame_index += 1
        self.step_position += (
            position.bpm * STEPS_PER_BEAT / (self.frame_rate * 60.0)
        )
        return position


class CompositionPlanner:
    """Generate a deterministic, musically coordinated MRT conditioning stream."""

    def __init__(self, session_id: str, *, settings: PlannerSettings | None = None):
        self.seed = stable_seed(session_id)
        self.tonic = KEY_ROOTS[self.seed % len(KEY_ROOTS)]
        self.clock = CompositionClock()
        initial = self._normalize_settings(settings or PlannerSettings())
        self._active_settings = initial
        self._pending_settings = initial
        self._last_step = -1
        self._previous_notes: frozenset[int] = frozenset()
        self._phrase_cache: dict[tuple, tuple[MotifNote, ...]] = {}
        self._voicing_cache: dict[tuple, tuple[tuple[int, ...], ...]] = {}

    @property
    def settings(self) -> PlannerSettings:
        """Most recently requested settings (possibly awaiting a step boundary)."""

        return self._pending_settings

    @property
    def active_settings(self) -> PlannerSettings:
        return self._active_settings

    @property
    def frame_index(self) -> int:
        return self.clock.frame_index

    @property
    def key_root(self) -> int:
        return self.tonic

    def configure(
        self,
        *,
        station: str | None = None,
        mood: str | None = None,
        bpm: float | None | object = ...,
        groove: float | None = None,
        intensity: float | None = None,
        melody_enabled: bool | None = None,
        drums_enabled: bool | None = None,
        strict_drums: bool | None = None,
        guide_mode: GuideMode | str | None = None,
    ) -> PlannerSettings:
        """Queue runtime controls without changing the clock or current phase.

        Changes become active together on the next sixteenth-note boundary.  A
        BPM of ``None`` restores the mood-sensitive automatic tempo; omitting
        ``bpm`` leaves the existing override untouched.
        """

        next_settings = self._pending_settings
        updates = {}
        if station is not None:
            updates["station"] = _station_name(station)
        if mood is not None:
            updates["mood"] = normalize_mood(mood)
        if bpm is not ...:
            updates["bpm"] = None if bpm is None else _clamp(float(bpm), BPM_MIN, BPM_MAX)
        if groove is not None:
            updates["groove"] = _clamp(groove, 0.0, 1.0)
        if intensity is not None:
            updates["intensity"] = _clamp(intensity, 0.0, 1.0)
        if melody_enabled is not None:
            updates["melody_enabled"] = bool(melody_enabled)
        if drums_enabled is not None:
            updates["drums_enabled"] = bool(drums_enabled)
        if strict_drums is not None:
            updates["strict_drums"] = bool(strict_drums)
        if guide_mode is not None:
            try:
                updates["guide_mode"] = GuideMode(guide_mode)
            except ValueError as exc:
                raise ValueError(f"unknown guide mode: {guide_mode}") from exc
        self._pending_settings = self._normalize_settings(
            replace(next_settings, **updates)
        )
        return self._pending_settings

    update = configure

    def plan(
        self, mood: str | None, frames: int
    ) -> list[PianoRollRun]:
        """Return run-length encoded ``(event, frames)`` conditioning."""

        events = self.plan_frames(mood, frames)
        runs: list[PianoRollRun] = []
        for event in events:
            if runs and runs[-1][0] == event:
                previous, length = runs[-1]
                runs[-1] = (previous, length + 1)
            else:
                runs.append((event, 1))
        return runs

    def plan_frames(
        self, mood: str | None, frames: int
    ) -> list[PianoRollEvent]:
        """Return one metadata-rich event per 40 ms model frame."""

        if frames <= 0:
            return []
        if mood is not None:
            self.configure(mood=mood)
        return [self._next_event() for _ in range(frames)]

    def section_at(self, bar: int) -> Section:
        cycle, bar_in_form = divmod(max(0, int(bar)), FORM_BARS)
        cursor = 0
        for spec in FORM:
            if bar_in_form < cursor + spec.bars:
                return Section(
                    name=spec.name,
                    cycle=cycle,
                    bar_in_form=bar_in_form,
                    bar_in_section=bar_in_form - cursor,
                    length_bars=spec.bars,
                )
            cursor += spec.bars
        raise AssertionError("form layout does not cover its declared length")

    def chord_at(
        self,
        mood: str,
        bar: int,
        *,
        station: str | None = None,
        intensity: float | None = None,
    ) -> Chord:
        mood = normalize_mood(mood)
        section = self.section_at(bar)
        station = _station_name(station or self._active_settings.station)
        intensity = (
            self._active_settings.intensity if intensity is None else _clamp(intensity, 0, 1)
        )
        degree = self._degree_at(mood, bar, station)
        tone_count = 2 if intensity < 0.35 else 3 if intensity < 0.72 else 4
        cache_key = (mood, station, section.cycle, tone_count)
        voicings = self._voicing_cache.get(cache_key)
        if voicings is None:
            previous: tuple[int, ...] = ()
            planned = []
            first_bar = section.cycle * FORM_BARS
            for form_bar in range(FORM_BARS):
                absolute_bar = first_bar + form_bar
                item_degree = self._degree_at(mood, absolute_bar, station)
                previous = self._chord_voicing(
                    mood, item_degree, intensity, previous=previous
                )
                planned.append(previous)
            voicings = tuple(planned)
            self._voicing_cache[cache_key] = voicings
        notes = voicings[section.bar_in_form]
        return Chord(degree=degree, symbol=ROMAN[mood][degree], notes=notes)

    def _degree_at(self, mood: str, bar: int, station: str) -> int:
        """Return a section-aware degree while preserving A/reprise identity."""
        section = self.section_at(bar)
        variants = PROGRESSIONS[mood]
        base_index = stable_seed(
            f"{self.seed}:{station}:{mood}:{section.cycle}:theme-a"
        ) % len(variants)
        if section.name == "theme_b":
            # B contrasts with A, but it is a deterministic neighboring option
            # rather than an unrelated progression drawn from a new section seed.
            shift = 1 + stable_seed(
                f"{self.seed}:{station}:{mood}:{section.cycle}:theme-b"
            ) % (len(variants) - 1)
            progression = variants[(base_index + shift) % len(variants)]
            index = section.bar_in_section
        else:
            progression = variants[base_index]
            # Intro previews A, breakdown revisits its second half, and reprise
            # returns to the exact eight-bar A progression.
            index = (
                section.bar_in_section + 4
                if section.name == "breakdown"
                else section.bar_in_section
            )
        return progression[index % len(progression)]

    def phrase_degrees(
        self, mood: str, phrase_index: int
    ) -> tuple[int | None, ...]:
        """Return a 32-step compatibility/debug view of one varied motif."""

        settings = replace(self._active_settings, mood=normalize_mood(mood))
        notes = self._phrase_notes(settings, max(0, int(phrase_index)))
        result: list[int | None] = [None] * PHRASE_STEPS
        for note in notes:
            for step in range(note.start, min(PHRASE_STEPS, note.start + note.duration)):
                result[step] = note.degree
        return tuple(result)

    def _next_event(self) -> PianoRollEvent:
        old_bpm = self._active_settings.resolved_bpm
        position = self.clock.position(old_bpm)
        new_step = position.absolute_step != self._last_step
        if new_step:
            # Controls switch as one transaction at a grid boundary.  Recompute
            # metadata with the new BPM, but retain the fractional clock phase.
            self._active_settings = self._pending_settings
            position = self.clock.position(self._active_settings.resolved_bpm)
            self._last_step = position.absolute_step

        settings = self._active_settings
        section = self.section_at(position.bar)
        chord = self.chord_at(
            settings.mood,
            position.bar,
            station=settings.station,
            intensity=settings.intensity,
        )
        melody_note = self._melody_at(settings, section, position, chord)

        guided = (
            settings.guide_mode is GuideMode.GUIDED and settings.melody_enabled
        )
        if guided:
            lead = (melody_note,) if melody_note is not None else ()
            harmony = self._harmony_at(settings, section, position, chord)
            desired = frozenset((*harmony, *lead))
        else:
            desired = frozenset()
            melody_note = None

        onsets = desired - self._previous_notes
        releases = self._previous_notes - desired
        sustains = desired & self._previous_notes
        tokens = [-1] * 128
        for note in releases:
            tokens[note] = 0
        for note in sustains:
            tokens[note] = 1
        for note in onsets:
            tokens[note] = 2

        # Once explicit releases have been sent, a disabled guide becomes truly
        # unconstrained rather than emitting 128 note-off tokens forever.
        self._previous_notes = desired
        drum = self._drum_at(settings, section, position, new_step)
        key_mode = "minor" if settings.mood == "somber" else "major"
        event = PianoRollEvent(
            tokens=tuple(tokens),
            drum=drum,
            position=position,
            mood=settings.mood,
            key=f"{PITCH_NAMES[self.tonic % 12]} {key_mode}",
            section=section,
            chord=chord,
            melody_note=melody_note,
            active_notes=tuple(sorted(desired)),
            onset_notes=tuple(sorted(onsets)),
            released_notes=tuple(sorted(releases)),
        )
        self.clock.advance(position.bpm)
        return event

    def _chord_voicing(
        self,
        mood: str,
        degree: int,
        intensity: float,
        *,
        previous: tuple[int, ...] = (),
    ) -> tuple[int, ...]:
        """Choose a compact jazz voicing with minimum motion from the last bar."""
        scale = MODE_SCALES[mood]
        if intensity < 0.35:
            # Bass supplies the root; third and seventh carry the harmony
            # clearly without turning a quiet station into a low block chord.
            offsets = (2, 6)
        elif intensity < 0.72:
            offsets = (0, 2, 6)
        else:
            offsets = (0, 2, 4, 6)

        pitch_classes = tuple(
            _scale_pitch(self.tonic, scale, degree + offset) % 12
            for offset in offsets
        )
        choices = tuple(
            tuple(pitch for pitch in range(50, 73) if pitch % 12 == pitch_class)
            for pitch_class in pitch_classes
        )
        candidates = set()
        for raw in itertools.product(*choices):
            candidate = tuple(sorted(raw))
            if len(set(candidate)) != len(offsets) or candidate[-1] - candidate[0] > 18:
                continue
            candidates.add(candidate)
        if not candidates:
            raise RuntimeError("could not build a compact chord voicing")

        def score(candidate: tuple[int, ...]):
            center = sum(candidate) / len(candidate)
            if previous:
                if len(previous) == len(candidate):
                    motion = sum(abs(a - b) for a, b in zip(previous, candidate))
                else:
                    motion = sum(min(abs(note - old) for old in previous) for note in candidate)
            else:
                motion = 0
            common = len(set(previous) & set(candidate))
            return (
                motion * 4 - common * 3 + abs(center - 60.0),
                candidate[-1] - candidate[0],
                candidate,
            )

        return min(candidates, key=score)

    def _bass_pitch(self, mood: str, degree: int) -> int:
        pitch = _scale_pitch(self.tonic, MODE_SCALES[mood], degree)
        while pitch > 47:
            pitch -= 12
        while pitch < 32:
            pitch += 12
        return pitch

    @staticmethod
    def _groove_delay(groove: float) -> float:
        # Delay eighth-note offbeats by at most ~70ms at typical lofi tempi.
        # Fractional step timing survives until the 25fps model clock instead
        # of moving a pickup wholesale onto the following downbeat.
        return max(0.0, groove - 0.45) * 0.65

    @staticmethod
    def _inside_window(position: float, start: float, duration: float) -> bool:
        return start <= position < start + duration

    def _harmony_at(
        self,
        settings: PlannerSettings,
        section: Section,
        position: ClockPosition,
        chord: Chord,
    ) -> tuple[int, ...]:
        """Articulate bass and comping instead of holding a bar-long block."""
        step = position.step_in_bar + position.phase_in_step
        delay = self._groove_delay(settings.groove)
        intensity = settings.intensity

        if section.name == "breakdown":
            comp_windows = [(0.0, 2.0)]
            if intensity >= 0.58:
                comp_windows.append((8.0, 1.3))
        elif section.name == "intro":
            comp_windows = [(0.0, 2.2)]
            if intensity >= 0.50:
                comp_windows.append((8.0, 1.4))
        else:
            comp_windows = [(0.0, 2.0), (12.0, 1.5)]
            if intensity >= 0.30:
                comp_windows.insert(1, (6.0 + delay, 1.25))
            if intensity >= 0.62:
                comp_windows.append((10.0 + delay, 0.9))

        bass_windows = [(0.0, 3.0)]
        if section.name != "breakdown" and intensity >= 0.45:
            bass_windows.append((8.0, 2.0))

        sounding = []
        if any(self._inside_window(step, start, length) for start, length in comp_windows):
            sounding.extend(chord.notes)
        if any(self._inside_window(step, start, length) for start, length in bass_windows):
            sounding.append(self._bass_pitch(settings.mood, chord.degree))
        return tuple(sorted(set(sounding)))

    def _melody_at(
        self,
        settings: PlannerSettings,
        section: Section,
        position: ClockPosition,
        chord: Chord,
    ) -> int | None:
        phrase_index, phrase_step = divmod(position.absolute_step, PHRASE_STEPS)
        phrase_position = phrase_step + position.phase_in_step
        notes = self._phrase_notes(settings, phrase_index)
        for note in notes:
            delay = self._groove_delay(settings.groove) if note.start % 4 == 2 else 0.0
            if note.start % 4 == 3:
                delay = self._groove_delay(settings.groove) * 0.45
            start = note.start + delay
            gate = 0.66 + settings.groove * 0.14
            if settings.mood == "lively":
                gate -= 0.06
            duration = max(0.65, note.duration * gate)
            if not self._inside_window(phrase_position, start, duration):
                continue
            scale = MODE_SCALES[settings.mood]
            spec = next(item for item in FORM if item.name == section.name)
            degree = self._harmonized_melody_degree(note, chord, scale)
            # Keep the lead above the mid-register comping so a melodic release
            # cannot be hidden by the same pitch remaining active in a chord.
            pitch = _scale_pitch(self.tonic + 36 + spec.register, scale, degree)
            return max(0, min(127, pitch))
        return None

    def _harmonized_melody_degree(
        self,
        note: MotifNote,
        chord: Chord,
        scale: tuple[int, ...],
    ) -> int:
        """Anchor structural notes to the chord; keep short weak tones mobile."""
        nominal = note.degree
        chord_classes = {
            _scale_pitch(self.tonic, scale, chord.degree + offset) % 12
            for offset in (0, 2, 4)
        }
        nominal_class = _scale_pitch(self.tonic, scale, nominal) % 12
        nearest_semitones = min(
            min((nominal_class - item) % 12, (item - nominal_class) % 12)
            for item in chord_classes
        )
        structural = note.start % 4 == 0 or note.start >= PHRASE_STEPS - 4
        avoid_long_semitone = note.duration >= 2 and nearest_semitones == 1
        if not structural and not avoid_long_semitone:
            return nominal

        candidates = [
            chord.degree + offset + octave * 7
            for octave in range(-2, 4)
            for offset in (0, 2, 4)
            if 0 <= chord.degree + offset + octave * 7 <= 10
        ]
        return min(candidates, key=lambda value: (abs(value - nominal), value))

    def _phrase_notes(
        self, settings: PlannerSettings, phrase_index: int
    ) -> tuple[MotifNote, ...]:
        section = self.section_at((phrase_index * PHRASE_BARS))
        family = "theme_a" if section.name == "reprise" else section.name
        family_phrase = section.bar_in_section // PHRASE_BARS
        key = (
            settings.mood,
            section.cycle,
            family,
            family_phrase,
            settings.station,
            round(settings.intensity, 3),
        )
        cached = self._phrase_cache.get(key)
        if cached is not None:
            return cached

        notes = list(BASE_MOTIFS[settings.mood])
        # Reprise deliberately reuses A's motif seed and density. Arrangement
        # context still differs through harmony articulation and audio history.
        section_spec = next(item for item in FORM if item.name == family)
        rng = random.Random(stable_seed(f"{self.seed}:{key}"))

        # Develop the motif in a four-phrase statement/variation/answer/cadence
        # cycle.  Strong anchors and the final tonic remain recognizable.
        role = family_phrase % 4
        varied = []
        for index, note in enumerate(notes):
            degree = note.degree
            if 0 < index < len(notes) - 1:
                mutation_chance = 0.08 + 0.18 * settings.intensity
                if role == 1 and index % 3 == 1:
                    degree += 1 if rng.random() < 0.5 else -1
                elif role == 2 and index % 3 == 2:
                    degree += 2
                elif rng.random() < mutation_chance:
                    degree += 1 if rng.random() < 0.5 else -1
            varied.append(MotifNote(note.start, note.duration, max(0, min(7, degree))))
        notes = varied

        density = section_spec.density * (0.68 + 0.64 * settings.intensity)
        if density < 1.0:
            kept = [notes[0]]
            kept.extend(
                note
                for index, note in enumerate(notes[1:-1], start=1)
                if rng.random() < density or index % 3 == 0
            )
            kept.append(notes[-1])
            notes = kept
        elif density > 1.0:
            candidates = (5, 14, 20, 27)
            occupied = {note.start for note in notes}
            for start in candidates:
                if start in occupied or rng.random() > min(0.7, density - 0.75):
                    continue
                previous = max((n for n in notes if n.start < start), key=lambda n: n.start)
                notes.append(MotifNote(start, 1, max(0, previous.degree - 1)))

        notes.sort(key=lambda note: note.start)
        cleaned: list[MotifNote] = []
        for index, note in enumerate(notes):
            next_start = notes[index + 1].start if index + 1 < len(notes) else PHRASE_STEPS
            duration = max(1, min(note.duration, next_start - note.start))
            cleaned.append(MotifNote(note.start, duration, note.degree))

        # Every phrase gets a real cadence; variation happens before it.
        last = cleaned[-1]
        cleaned[-1] = MotifNote(last.start, min(last.duration, PHRASE_STEPS - last.start), 0)
        result = tuple(cleaned)
        self._phrase_cache[key] = result
        return result

    def _drum_at(
        self,
        settings: PlannerSettings,
        section: Section,
        position: ClockPosition,
        new_step: bool,
    ) -> int:
        if not settings.drums_enabled:
            return 0
        if not settings.strict_drums:
            # Drum-unconditional is the supported "drums on" behavior. The
            # model decides the pattern from its style and audio history.
            return -1
        if not new_step:
            return 0

        step = position.step_in_bar
        # Experimental strict rhythm: because the input is only a binary hit,
        # it cannot choose kick/snare/hat. Explicit zeroes are essential here;
        # masking non-attacks would permit arbitrary hits between anchors.
        attacks = {0, 4, 8, 12}
        if settings.intensity > 0.42:
            attacks.add(10)
        if settings.intensity > 0.72:
            attacks.update((2, 6, 14))
        if settings.groove > 0.68:
            attacks.discard(8)
            attacks.add(9)
        if section.name in ("intro", "breakdown"):
            attacks.intersection_update((0, 8, 12))
        return 1 if step in attacks else 0

    @staticmethod
    def _normalize_settings(settings: PlannerSettings) -> PlannerSettings:
        try:
            mode = GuideMode(settings.guide_mode)
        except ValueError as exc:
            raise ValueError(f"unknown guide mode: {settings.guide_mode}") from exc
        return PlannerSettings(
            station=_station_name(settings.station),
            mood=normalize_mood(settings.mood),
            bpm=(
                None
                if settings.bpm is None
                else _clamp(settings.bpm, BPM_MIN, BPM_MAX)
            ),
            groove=_clamp(settings.groove, 0.0, 1.0),
            intensity=_clamp(settings.intensity, 0.0, 1.0),
            melody_enabled=bool(settings.melody_enabled),
            drums_enabled=bool(settings.drums_enabled),
            strict_drums=bool(settings.strict_drums),
            guide_mode=mode,
        )


def expand_runs(runs: Iterable[PianoRollRun]) -> list[PianoRollEvent]:
    """Expand run-length encoded events; useful for adapters and tests."""

    result = []
    for event, frames in runs:
        if frames < 0:
            raise ValueError("run length cannot be negative")
        result.extend([event] * frames)
    return result


__all__ = [
    "BEATS_PER_BAR",
    "BPM_MAX",
    "BPM_MIN",
    "Chord",
    "ClockPosition",
    "CompositionClock",
    "CompositionPlanner",
    "FORM_BARS",
    "FRAME_RATE",
    "GuideMode",
    "KEY_ROOTS",
    "MODE_SCALES",
    "MOOD_BPMS",
    "PHRASE_STEPS",
    "PianoRollEvent",
    "PianoRollRun",
    "PlannerSettings",
    "SCALES",
    "STEPS_PER_BAR",
    "STEPS_PER_BEAT",
    "Section",
    "expand_runs",
    "normalize_mood",
    "stable_seed",
]
