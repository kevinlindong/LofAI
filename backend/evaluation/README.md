# MRT2 render and listening evaluation

This package renders a deterministic matrix through lofAI's real
`MRTEngine.generate()` path, writes ordinary PCM16 WAV files, records every
requested and observed setting in JSON/CSV, calculates conservative signal
diagnostics, and produces a dependency-free blind pairwise listening page.
It is separate from the server and does not alter live behavior.

The signal measurements are **not music-quality scores**. RMS, a gated
loudness proxy, clipping, bandwidth, spectral flatness, onset activity, and
short-lag repetition can expose broken or surprising audio. Only blinded
listening can answer whether somebody wants to hear the music.

## Quick start

Run these commands from the repository root with the backend virtual
environment installed:

For one quality-first offline candidate, use the 2.4B base model (or select
`mrt2_small`) at a pinned full 12-codebook depth. This mode deliberately accepts
only 30–90 second excerpts:

```bash
./venv/bin/python -m backend.evaluation candidate \
  --model-size mrt2_base \
  --duration-seconds 60 \
  --seed 20260903 \
  --output evaluation-runs/base-candidate
```

This is useful on hardware where `mrt2_base` cannot run live: output quality is
kept at full depth and rendering simply takes longer than playback. The
convenience mode uses the live `dusty-beats` station, production sampling,
and one bar of preroll by default.

For several reproducible takes of the same style/settings, followed by signal
triage and actual blind review:

```bash
./venv/bin/python -m backend.evaluation multi-take \
  --model-size mrt2_base \
  --duration-seconds 60 \
  --takes 6 \
  --seed-start 20260903 \
  --shortlist-count 3 \
  --output evaluation-runs/base-multi-take
```

By default these are whole-take variations: both the decoder performance and
the deterministic composition seed change. To compare decoder performances of
one fixed composition instead, add `--composition-seed 4242`.

`multi-take` writes `signal_shortlist.json`. This is explicitly a
non-authoritative heuristic: it can only deprioritize broad signal risks such
as clipping, extreme level, a compound noise signature, severe bandwidth loss,
or an almost exact short energy-envelope recurrence. It does not reward or
measure musicality. Listen blind before selecting a take. All convenience
settings can be overridden; run `... candidate --help` or
`... multi-take --help` for station/custom prompt, optional audio reference, BPM, groove,
intensity, melody/drum modes, sampling/CFG, and quantization flags. A custom
prompt is used with `--station custom`; named stations resolve the same short
prompt and optional `MRT_STYLE_REFERENCE_DIR` WAV as the live application.

The same signal triage can be rebuilt without rendering:

```bash
./venv/bin/python -m backend.evaluation shortlist \
  --manifest evaluation-runs/base-multi-take/manifest.json \
  --count 3
```

For a controlled parameter matrix, start from the checked-in JSON:

```bash
# Validate the config, enumerate exact case IDs, and estimate runtime.
./venv/bin/python -m backend.evaluation plan \
  --config backend/evaluation/example_config.json

# Render the matrix and build the blind report.
./venv/bin/python -m backend.evaluation render \
  --config backend/evaluation/example_config.json \
  --output evaluation-runs/mrt2-baseline

# Resume only missing, errored, or hash-mismatched cases after an interruption.
./venv/bin/python -m backend.evaluation render \
  --config backend/evaluation/example_config.json \
  --output evaluation-runs/mrt2-baseline \
  --resume
```

The render command creates:

- `wav/*.wav`: unprocessed 48 kHz stereo PCM16 excerpts.
- `manifest.json`: full nested configuration, provenance, adapter facts,
  SHA-256 hashes, timings, and metrics.
- `manifest.csv`: the same comparison fields flattened for analysis.
- `listening_report.html`: the static A/B preference task.
- `blind/*.wav`: randomly renamed copies used by that task.
- `blind_key.json`: private mapping from blind clips/trials to case IDs.

Serve the listening folder locally so every browser can seek audio reliably:

```bash
./venv/bin/python -m http.server 8081 --directory evaluation-runs/mrt2-baseline
```

Then open `http://localhost:8081/listening_report.html`. Give listeners the
report and `blind/` audio but hold back `blind_key.json`. The page downloads a
small response JSON when the listener finishes. Decode it with:

```bash
./venv/bin/python -m backend.evaluation decode \
  --key evaluation-runs/mrt2-baseline/blind_key.json \
  --results ~/Downloads/lofai-listening-RESULT.json \
  --output evaluation-runs/mrt2-baseline/listener-01.csv
```

To rebuild a report with a different deterministic trial order or cap:

```bash
./venv/bin/python -m backend.evaluation report \
  --manifest evaluation-runs/mrt2-baseline/manifest.json \
  --seed 271828 \
  --max-pairs 40
```

To inspect an unrelated PCM16 WAV with the same diagnostics:

```bash
./venv/bin/python -m backend.evaluation analyze path/to/clip.wav \
  --output path/to/clip.metrics.json
```

## Matrix configuration

Every decoder seed, style/scenario, preset, note mode, drum mode, and codebook
depth is crossed. The included live-start config covers all four stations,
three paired decoder/composition seeds, the old raw settings and current
production settings, guided/unconditioned notes, and 10/12 codebooks:
`3 × 4 × 2 × 2 × 1 × 2 = 96` excerpts. Durations must be multiples of MRT2's
40 ms frame. General matrix durations have no 90-second limit, so a config can
cover the planner's complete 32-bar form. The 30–90 second bound applies only
to the quick `candidate` and `multi-take` commands.

The example deliberately uses zero preroll because listeners hear the live
session start. For longer-form musical comparisons, copy it and prefer
`warmup_bars`. That generates preroll in the same
seeded state and trims at the first model frame in the requested bar, rather
than cutting into an arbitrary beat or note. `warmup_seconds` remains available
for transient experiments. The manifest records the actual warmup frames and
fractional step phase at the listening start; neither mode resets or splices
the model state.

A style entry may name a production `station`, in which case its prompt,
station BPM/groove/intensity, instrument/mood, and optional local audio
reference are resolved through the live style registry. Alternatively use
`station: "custom"` with explicit audible `prompt` tags and controls. Resolved
values, the absolute reference path, reference SHA-256, and the configured
text/audio blend all become part of the case identity and manifest. A changed
reference fails the render rather than silently changing an experiment.

Presets explicitly set temperature, top-k, all three CFG values, retained
MusicCoCa style levels, and `sampling_mode`. `raw` applies those values exactly
for a parameter experiment. `production` follows `Session._sampling_for`,
which now keeps stochastic settings stable while intensity acts through the
score and arrangement; effective values are recorded separately in JSON/CSV.
The example's production values come from Magenta's current Jam/AU shared
settings; they are a useful baseline, not a claimed optimum.

Supported note modes are:

- `masked`: keeps every pitch masked. In MRT2 this means the model is
  unconstrained; it does **not** force silence. Drum mode remains independent,
  so a masked-vs-guided note comparison is not accidentally a drum comparison.
- `app_melody`: uses lofAI's deterministic production composition planner,
  including exact onset/sustain/release piano-roll tokens and synchronized drum
  intent. The same seed/style gets the same guide across preset and codebook
  comparisons.
- `constant`: supplies one MIDI pitch for the whole render; include a
  `midi_note` integer from 0 through 127. It sends token `2` once and token `1`
  afterward, not the permissive Auto-Strum token `3`.

Supported drum modes are `planned` (production parity: masked/model-decided when
drums are enabled, explicit zero when disabled), `masked` (model decides), and
`off` (explicit zero). `strict` is an experimental binary 1/0 bar pulse for an
A/B test; it is not the live default because MRT2's authors describe direct
drum-hit control as impractical and use this channel as drums on/off. The
style's `drums` boolean is carried into `planned` and `strict`.

Codebook depths from 8 through 12 are accepted. Values below the live app's
normal floor are allowed here specifically for controlled listening tests.

## Reproducibility and interpretation

Each case gets a stable ID derived from all audio-affecting inputs, and every
WAV is hashed. `seeds` are installed into the initial decoder state used by
the eager fast path. By default each is also the composition seed; set one
top-level `composition_seed` to isolate decoder variation. Runs reject `mlxfn`
and intentionally fail if MRTEngine cannot expose the requested codebook depth
through its eager fast path. For byte-identical reruns, keep the Git
revision, Magenta/MLX versions, model checkpoint, model size, quantization,
hardware/backend, and config unchanged; the manifest captures the available
software provenance and observed adapter settings.

The normal matrix pairwise report compares clips only within the same style and
seed, then randomizes clip aliases, trial order, and A/B side. It may compare
preset, note-mode, and codebook changes. `multi-take` instead compares seeds
within one style so listeners can pick performances. The standalone `report`
command supports the same behavior with `--compare-across-seeds`. Use multiple
listeners and seeds, randomize headphone/order effects, and analyze preferences
by one controlled dimension where possible.

Metric meanings:

- `gated_loudness_proxy_dbfs` uses 400 ms overlapping blocks and LUFS-like
  absolute/relative gates, but deliberately omits K-weighting. It is not LUFS.
- `spectral_rolloff_95_hz` and `high_band_energy_fraction` describe bandwidth.
- `spectral_flatness` rises for noise-like spectra but can legitimately rise
  for percussion or tape noise.
- `onset_proxy_per_second` counts spectral-flux peaks, not musical notes.
- `repetition_peak_correlation` finds the strongest 0.5–8 s recurrence in a
  50 ms energy envelope. A steady groove can legitimately score highly.

No metric is thresholded or aggregated into a quality verdict.

## Tests (no model load)

The tests use a deterministic mock adapter and generated waveforms. They never
import or initialize Magenta RealTime:

```bash
./venv/bin/python -m unittest discover \
  -s backend/evaluation/tests \
  -p 'test_*.py' \
  -v
```

The same suite also works from the backend directory, matching the existing
backend test convention:

```bash
cd backend
../venv/bin/python -m unittest discover -s evaluation/tests -p 'test_*.py' -v
```
