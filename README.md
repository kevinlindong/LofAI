# LofAI

An AI generated lofi music player. Every listener gets their own endless stream, generated live, and switching stations bends the music as it plays.

## Screenshot:

<img width="882" alt="The lofAI panel: circular dot visualiser, task list, focus timer, and the cat" src="lofAI.png">

## How it Works:

- Google DeepMind's Magenta RealTime 2 (`mrt2_small`) is the live performer and renderer. Every 40ms it samples SpectroStream tokens, decodes 48kHz stereo audio, and carries a separate recurrent state for each listener
- The live path is prompt-first: one concise MusicCoCa station embedding guides MRT2 while its own audio history supplies the musical continuity. Piano-roll generation, bar clocks, and per-frame score churn stay out of the hot loop
- Four curated stations provide coherent style targets. The only musical override is drums on/off; station changes start on the next model chunk and glide over 320ms
- **New take** resets the recurrent model state and sampling seed together, then prevents old in-flight audio from leaking into the new variation
- Raw PCM travels over a WebSocket into one AudioWorklet read cursor. A slow loudness normalizer, transparent limiter, and output ceiling make station changes consistent without pumping or clipping
- Four-bit weights, GPU keepalive, short startup chunks, and adaptive 10-12 layer codec output keep the M1 Air ahead of playback. The browser starts from a sub-second reservoir instead of hiding deficits behind seconds of buffering
- The old symbolic composer remains available only to the offline evaluation harness for matched, fixed-seed comparisons. Signal metrics catch broken audio; people decide whether the result is good music

See [the streaming design note](docs/MUSIC_STREAMING.md) for the full MRT2 pipeline, bottleneck analysis, and local benchmarks.

## Languages and Frameworks:

- **Backend**: Python, FastAPI, WebSockets
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, Canvas 2D
- **AI**: Google DeepMind's Magenta RealTime 2 live music model (MLX)

## The Interface:

The whole page is a dot matrix display, set in a real 5x7 dot matrix face. Dark is an LED sign; light is a printout, ink struck onto paper. Both themes drive the same eight-step dot ramp (`--dot-0` to `--dot-7`), which is what the two canvases paint with.

- The **visualiser** is a polar dot matrix. Dots sit at a constant pitch rather than a constant angle - each ring holds as many as fit around it - so the field stays even instead of fanning into rays at the rim. The spectrum is mirrored and resampled around the circle, and each column lights from the inside out, so the wave rides the music with a bright crest and a peak-hold dot that falls behind it
- The **cat** is a tamagotchi that watches you work. It blinks on its own, nods on the beat while the music plays, narrows its eyes while the focus timer runs, perks an ear when you add a task, hops with sparkles when you finish one - three hops if that was the last one - and nods off if you leave it alone. The scene is drawn cell by cell every frame in `frontend/lib/pet-scene.ts`, which is also where you would go to redraw it
- Both canvases read their colours from CSS variables and watch the root element for a theme flip, so switching themes repaints them without a reload
- Everything honours `prefers-reduced-motion`: the animation loops never start, and both canvases draw one resting frame instead

## Requirements:

Live local generation needs an **Apple Silicon Mac**. `mrt2_small` is the practical interactive model; `mrt2_base` is intended for higher-quality offline evaluation or substantially faster hardware. An 8GB M1 is close to the real-time boundary, so use `GET /health` rather than assuming a particular model or codec depth will keep up on every machine.

Python 3.11 or 3.12, and about 3GB of disk for the model assets.

## Setup:

No API key needed — the model runs locally.

1. Install dependencies:
```bash
# backend
python3.12 -m venv venv        # or python3.11
source venv/bin/activate
pip install -r backend/requirements.txt

# frontend
cd frontend
npm install
```

2. Download the model (about 3GB, first run only):
```bash
mrt models init                      # MusicCoCa + SpectroStream
mrt checkpoints download mrt2_small  # the streaming model
```

`./backend_start.sh` does both of these for you if you skip this step. Assets land in `~/Documents/Magenta`; set `MAGENTA_HOME` to put them elsewhere.

## Run:

```bash
# Option 1: Use start script (runs both)
./start.sh

# Option 2: Run separately
./backend_start.sh  # Terminal 1
./frontend_start.sh # Terminal 2
```

`./start.sh` is the foreground supervisor for the complete application. Leave
it running while you use lofAI; Ctrl+C, terminal close, or a service failure
shuts down the backend, frontend, model inference, Next.js workers, and log
follower together. If you choose the two-terminal option, Ctrl+C each of those
foreground commands when you are done.

The frontend launcher serves a production build by default. For Next.js hot
reload while developing, run `LOFAI_FRONTEND_MODE=development ./frontend_start.sh`.

Backend: http://localhost:8000
Frontend: http://localhost:3000

The backend binds to loopback by default. To listen from another device, set
`LOFAI_BACKEND_HOST=0.0.0.0`, point `NEXT_PUBLIC_BACKEND_HOST` at the Mac, and
add the frontend origin to the comma-separated `LOFAI_ALLOWED_ORIGINS`. Keep
the default loopback binding unless LAN access is intentional.

The HTTP server starts immediately while the model loads, checks decoded audio,
and calibrates itself; the play button reports "warming up the model" until it
is ready. Mapped MusicCoCa embeddings are cached under
`~/Library/Caches/lofai/embeddings`, so later starts skip the text-encoder work.
Set `MRT_EMBEDDING_CACHE=` to disable the cache or point it elsewhere.

## Tuning:

The backend runs every session on one thread, so how many people can listen at once depends on how much faster than real time the model generates. `GET /health` reports `renderRealtimeFactor` for the model and `perSessionRealtimeFactor` after dividing that capacity among active listeners. The latter needs to stay above 1.0. A machine with headroom can raise `MRT_MAX_SESSIONS`; admission is still bounded by measured throughput.

| Variable | Default | What it does |
|---|---|---|
| `MRT_MODEL_SIZE` | `mrt2_small` | `mrt2_small` or `mrt2_base` |
| `MRT_MAX_SESSIONS` | `1` | Concurrent streams. Extra listeners queue for a slot |
| `LOFAI_BACKEND_HOST` | `127.0.0.1` | Backend bind address; use `0.0.0.0` only for intentional LAN access |
| `LOFAI_ALLOWED_ORIGINS` | local frontend origins | Comma-separated browser origins allowed to open the music WebSocket |
| `MRT_TARGET_RTF` | `1.18` | How much faster than real time the auto-tuner aims to render. Raising it buys margin by spending audio detail |
| `MRT_TEMPERATURE` | `1.0` | Sampling randomness; matches the upstream native live runner |
| `MRT_TOP_K` | `100` | Candidate token pool; matches the upstream native live runner |
| `MRT_CFG_MUSICCOCA` | `3.0` | MusicCoCa style guidance strength |
| `MRT_CFG_NOTES` | `5.0` | Upstream-compatible MIDI guidance strength; live generation sends no MIDI |
| `MRT_CFG_DRUMS` | `1.0` | Drum guidance strength |
| `MRT_STYLE_TOKEN_LEVELS` | `6` | Coarse MusicCoCa RVQ levels retained; the fine tail is masked for stable live steering |
| `MRT_EMBEDDING_CACHE` | `~/Library/Caches/lofai/embeddings` | Persistent mapped-style cache, removing several seconds from later startups |
| `MRT_STYLE_REFERENCE_DIR` | unset | Optional directory of station WAV files named `<station>.wav` |
| `MRT_AUDIO_STYLE_BLEND` | `0.75` | Weight of reference-audio style versus its station text prompt |
| `MRT_CONDITIONING_CACHE_SIZE` | `128` | Maximum cached style/drum conditioning bundles |
| `MRT_CODEBOOKS` | auto | Pins the codebook count and turns the auto-tuner off |
| `MRT_MIN_CODEBOOKS` | `10` | Listener-facing codec quality floor (accepted range 8-12; lowering it is an explicit quality tradeoff) |
| `MRT_BITS` | `4` | Weight quantisation: `8`, `4`, or `0` for full precision. Four-bit is the measured live default on an 8GB M1 |
| `MRT_MLX_CACHE_MB` | `384` | Limit for reusable MLX buffers after calibration; prevents cache pressure and swap jitter on 8GB Macs |
| `MRT_FAST_SAMPLER` | `1` | Slice each codebook's valid logits before top-k sampling. Set `0` to use Magenta's generic sampler |
| `MRT_CHUNK_FRAMES` | `10` | Steady frames per model call (400ms). Larger saves a little pipeline overhead but delays controls and transport |
| `MRT_FIRST_CHUNK_FRAMES` | `8` | First burst size (320ms), followed by the steady chunk size |
| `MRT_LOOKAHEAD_SECONDS` | `0.4` | Server-side generated lead; lower makes station changes land sooner but leaves less scheduling cushion |
| `MRT_STYLE_RAMP_SECONDS` | `0.32` | How long a station change takes to fully land |
| `MRT_STYLE_STEP_FRAMES` | `2` | How finely a chunk is split while a station change is gliding |
| `MRT_SESSION_TTL` | `300` | How long a paused session keeps its state |
| `MRT_BACKEND` | `python` | Runs the checkpoint eagerly. `mlxfn` currently falls back to `python` because its output/seed path is not safe |

`GET /health` reports whether the model is loaded, how many sessions are active or queued, which codebook count the tuner has settled on, and per session its `realtimeFactor` and `gaps`.

## Musical quality and real time:

MRT2 is best treated as a responsive performer, not as a conventional prompt-to-finished-song service. The live stream now lets the model continue its own recurrent audio state instead of forcing a synthetic score into every frame. That removes a large control surface, makes the implementation easier to reason about, and sounded less rigid in local comparison. The former 32-bar composer is still available for controlled offline experiments, where latency is irrelevant and its musical value can be judged honestly.

Style prompts are deliberately short and concrete. A long list of genre, production, mood, and instrumentation adjectives can dilute MusicCoCa conditioning rather than improve it. A station may also have a reference WAV, which is embedded through the same native MusicCoCa path and blended with its text identity. The drum channel follows MRT2's supported on/off use: enabled leaves the model free to create a style-appropriate beat, while disabled explicitly requests drumless audio; a strict 1/0 pulse remains available only in the evaluation harness.

MRT2 samples one 40ms frame and up to 12 residual audio-codec layers at a time. More layers improve fidelity but cost time and also affect later recurrent state. Production defaults to four-bit weights and adaptive 10–12-layer rendering, targeting 1.18x real time. It refuses to hide a sustained render deficit behind a larger buffer: each active session must remain above a `realtimeFactor` of 1.0. The higher-quality evaluation path pins all 12 layers and can run slower than real time.

The eager MLX path pipelines frame evaluation and samples only the valid logits for each codebook. Post-calibration cache limits prevent MLX reusable buffers from crowding an 8GB machine into swap. The exported `mlxfn` backend remains disabled by default because locally exported and published graphs have produced invalid/noise-like decoding with the supported MLX versions.

Playback uses a single AudioWorklet cursor, so model chunks do not become browser scheduling seams. The browser starts after roughly 0.64-0.9 seconds when measured generation is healthy, uses a bounded eight-second ring, and keeps playback at 1.0x. Mastering changes gain slowly and catches only peaks, which avoids turning every kick into audible gain pumping.

Startup signal checks detect silence, clipping, corruption, DC, and obvious noise-like failure. They are guardrails, not musical-quality scores. Test candidate settings with the listening workflow below before changing production defaults.

## Quality evaluation:

The evaluation harness renders repeatable WAVs and a manifest for matched comparisons. Its default quality candidate uses `mrt2_base`, 12 codebooks, and 30–90 second excerpts. Multi-take mode can render several seeded performances and shortlist technically healthy takes, but the final decision remains a blind listening test.

See [`backend/evaluation/README.md`](backend/evaluation/README.md) for the current commands and output layout. Typical entry points are:

```bash
# Show all evaluation modes and flags
./venv/bin/python -m backend.evaluation --help

# Render one full-depth, fixed-seed quality candidate
./venv/bin/python -m backend.evaluation candidate \
  --model-size mrt2_base --duration-seconds 60 \
  --output evaluation-runs/base-candidate

# Validate, then render the checked-in comparison matrix
./venv/bin/python -m backend.evaluation plan \
  --config backend/evaluation/example_config.json
./venv/bin/python -m backend.evaluation render \
  --config backend/evaluation/example_config.json \
  --output evaluation-runs/mrt2-baseline
```

Generated reports include signal diagnostics such as loudness, clipping, DC, bandwidth, crest factor, and stereo correlation. Those metrics reject broken output and help compare mastering; they do not claim to measure melody, groove, coherence, or whether someone wants to keep listening.

## Stop:

```bash
./stop.sh
```

This is equivalent to Ctrl+C on the combined launcher and is safe to run more
than once. It stops each complete process group, waits for graceful shutdown,
and force-terminates a group only if it misses the shutdown deadline. PID files
are removed only after no processing remains.
