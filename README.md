# LofAI

An AI generated lofi music player. Every listener gets their own endless stream, generated live, and steering the mood or instrument bends the music as it plays.

## Screenshot:

<img width="882" alt="The lofAI panel: circular dot visualiser, task list, focus timer, and the cat" src="lofAI.png">

## How it Works:

- Backend runs Google DeepMind's Magenta RealTime 2 (`mrt2_small`), generating 48kHz stereo audio continuously instead of rendering fixed-length tracks
- A restrained per-session melody guide drives MRT2's native frame-level MIDI input: scale-bound phrases, rests, cadences, and subtle variations keep the lead melodic while the model still performs and arranges it
- Each listener gets their own session: its own model state, its own style, its own stream. One loaded model serves them all, because MRT2 keeps its streaming state outside the model
- Moving a slider retargets the style embedding and the stream glides onto the new setting over about a second, mid-music. There is no track boundary to wait for
- Audio arrives as raw PCM over a WebSocket and is played by an AudioWorklet that owns a ring buffer and one continuous read cursor, so the stream has no chunk boundaries to click at and a busy tab cannot stutter it
- The backend measures how fast it is actually rendering and keeps the highest safe codec depth without crossing the listener-facing quality floor — see [Keeping up with real time](#keeping-up-with-real-time)
- Pausing keeps your session warm, so unpausing picks the same music back up rather than starting over

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

Generation needs an **Apple Silicon Mac**. An 8GB M1 — the slowest thing this has been run on — stays around real time at the 10-layer quality floor; see [Keeping up with real time](#keeping-up-with-real-time) for the measured behavior. `mrt2_base` sounds better but needs considerably more machine — set `MRT_MODEL_SIZE=mrt2_base` if you have one.

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

`./start-backend.sh` does both of these for you if you skip this step. Assets land in `~/Documents/Magenta`; set `MAGENTA_HOME` to put them elsewhere.

## Run:

```bash
# Option 1: Use start script (runs both)
./start.sh

# Option 2: Run separately
./start-backend.sh  # Terminal 1
./start-frontend.sh # Terminal 2
```

`./start.sh` is the foreground supervisor for the complete application. Leave
it running while you use lofAI; Ctrl+C, terminal close, or a service failure
shuts down the backend, frontend, model inference, Next.js workers, and log
follower together. If you choose the two-terminal option, Ctrl+C each of those
foreground commands when you are done.

Backend: http://localhost:8000
Frontend: http://localhost:3000

The model loads in a few seconds and then spends another 5-15s embedding the style prompts; the play button reports "warming up the model" until both are done.
Mapped MusicCoCa embeddings are cached under `~/Library/Caches/lofai/embeddings`, so later starts skip that text-encoder work. Set `MRT_EMBEDDING_CACHE=` to disable the cache or point it elsewhere.

## Tuning:

The backend runs every session on one thread, so how many people can listen at once depends on how much faster than real time the model generates. `GET /health` reports a measured `realtimeFactor` per session: it needs to stay above 1.0 for each active listener. A machine with headroom can raise `MRT_MAX_SESSIONS`; watch `realtimeFactor` and `gaps` afterwards.

| Variable | Default | What it does |
|---|---|---|
| `MRT_MODEL_SIZE` | `mrt2_small` | `mrt2_small` or `mrt2_base` |
| `MRT_MAX_SESSIONS` | `1` | Concurrent streams. Extra listeners queue for a slot |
| `MRT_TARGET_RTF` | `1.15` | How much faster than real time the auto-tuner aims to render. Raising it buys margin by spending audio detail |
| `MRT_TEMPERATURE` | `1.0` | Sampling randomness. Higher is more varied; lower is more musically stable |
| `MRT_TOP_K` | `100` | Candidate token pool. Higher is more exploratory; lower is more repetitive |
| `MRT_CFG_MUSICCOCA` | `3.0` | Text-style guidance strength |
| `MRT_CFG_NOTES` | `5.0` | Strength of the native MIDI melody guide |
| `MRT_MELODY_GUIDE` | `1` | Keeps a scale-bound melodic phrase active. Set `0` only for an unguided A/B comparison |
| `MRT_STYLE_TOKEN_LEVELS` | `6` | Coarse MusicCoCa RVQ levels retained; the fine tail is masked for stable live steering |
| `MRT_EMBEDDING_CACHE` | `~/Library/Caches/lofai/embeddings` | Persistent mapped-style cache, removing several seconds from later startups |
| `MRT_CODEBOOKS` | auto | Pins the codebook count and turns the auto-tuner off |
| `MRT_MIN_CODEBOOKS` | `10` | Listener-facing codec quality floor (accepted range 8-12; lowering it is an explicit quality tradeoff) |
| `MRT_BITS` | `8` | Weight quantisation. `4` renders about 10% faster and halves the model's memory |
| `MRT_MLX_CACHE_MB` | `384` | Limit for reusable MLX buffers after calibration; prevents cache pressure and swap jitter on 8GB Macs |
| `MRT_FAST_SAMPLER` | `1` | Slice each codebook's valid logits before top-k sampling. Set `0` to use Magenta's generic sampler |
| `MRT_CHUNK_FRAMES` | `25` | Frames generated per model call (25 frames = 1s). Larger saves a little pipeline overhead but delays controls and transport |
| `MRT_LOOKAHEAD_SECONDS` | `2.0` | How far ahead of the wall clock to generate. This is the listener's reservoir; lower means slider changes land sooner, but leaves less cushion |
| `MRT_STYLE_RAMP_SECONDS` | `1.2` | How long a slider change takes to fully land |
| `MRT_STYLE_STEP_FRAMES` | `10` | How finely a chunk is split while a slider change is gliding |
| `MRT_SESSION_TTL` | `300` | How long a paused session keeps its state |
| `MRT_BACKEND` | `python` | Runs the checkpoint eagerly. `mlxfn` runs an exported graph — see below |

`GET /health` reports whether the model is loaded, how many sessions are active or queued, which codebook count the tuner has settled on, and per session its `realtimeFactor` and `gaps`.

## Keeping up with real time:

The stream is endless, so generation has to stay ahead of playback forever — and on an 8GB M1 the stock eager loop did not. `mrt2_small` rendered at **0.83x**, which no amount of buffering fixes: the reservoir just drains more slowly. The backend optimizes the full path rather than hiding that deficit behind a large delay.

**The model call pipelines.** MRT2 samples one 40ms frame at a time, and the library blocks on each one before building the next, so the CPU sits idle while the GPU works and vice versa. Handing each frame to `mx.async_eval` and only waiting on it one frame later overlaps the two. That is worth about 20% of wall clock — 48.4ms per frame down to 39.5ms, or 0.83x up to **1.05x** — and the audio is bit-for-bit identical to the blocking path, verified against a fixed seed.

**Sampling only touches valid logits.** MRT2 emits a shared 12,294-token vector at every depth step, but each residual codebook can select from one contiguous span of only 1,024 tokens. Magenta's generic path masks the rest and still sorts and samples the whole vector. The pinned 2.0.3 specialization slices first, preserving the same probability distribution. The measured gain ranges from negligible at some depths/thermal loads to roughly 7% at full depth. It is signature-guarded and falls back cleanly if disabled.

**Quality is a dial, with a hard default floor.** SpectroStream stacks 12 residual quantisers per frame and the depthformer samples them one after another, so each one costs about 1.1ms of a 40ms budget. At startup the backend measures every count from 12 down to 10 and keeps the highest one that clears `MRT_TARGET_RTF`. A normal startup reservoir dip can no longer lower it, and reported gaps cannot cross the floor. Codebook truncation also affects future model state, so the tuner keeps all 12 whenever measured headroom permits rather than treating it as a free codec-only reduction.

**Broken audio fails before playback.** Calibration retains at least two seconds of its throwaway PCM and checks structure, level, DC, clipping, and a compound white-noise signature. The engine does not become ready if that signal is silent, clipped, corrupt, or spectrally noise-like. This also guards alternate backends instead of trusting that a successful model load implies listenable audio.

**Melody is a control signal, not a hopeful adjective.** Each session owns a deterministic-but-varied four-bar motif at 78 BPM. It uses MRT2's 128-pitch piano-roll input to offer one tonal guide note at a time while leaving every other pitch masked, so the model is free to supply chords and accompaniment. Mood changes land on a musical step boundary, and seeded phrase variations prevent a short mechanical loop.

**MLX cannot crowd itself into swap.** Loading, quantizing, and calibration left roughly 1.66GB in MLX's reusable buffer cache on the 8GB M1; steady streaming used about 240MB. The post-calibration trim and 384MB limit reclaim roughly 1.3GB, removing memory-pressure spikes while leaving the live model and session state untouched.

With mapped prompts and the quality-oriented sampling defaults, the same 8GB M1 currently keeps 10 codebooks and measured about **1.00–1.09x** real time across cold-start runs. A machine that cannot reach the target at the floor stays at 10 rather than degrading the recurrent stream further.

**Playback stopped being the browser's spare time.** Audio used to be scheduled one `AudioBufferSourceNode` per chunk from a `setInterval`. That has two faults: the timer is on the main thread, so a busy tab schedules late, and each source resamples from its own phase, so any playback rate other than 1.0 leaves a discontinuity at every chunk seam. Now the chunks go into an `AudioWorklet` with one continuous fractional read cursor. Int16 PCM transfers directly from the WebSocket into an int16 ring, eliminating the main-thread conversion loop and halving ring memory; live resets fade before clearing, and paused contexts suspend after their fade.

That is the one piece that must never click, so it is tested for it. `npm test` covers steady playback, prebuffering, underrun/refill, rate changes, resampling, ring wraparound, and a live reset. The backend suites cover the tuner, worker-thread feedback, pause lead, TTL, style ramps, and the specialized sampler.

Because the backend reports its measured speed, the client sizes its reservoir to match rather than assuming the worst. Playback remains at exactly 1.0x, so buffer pressure never lowers the melody's pitch or adds slow wow.

The two canvases also stopped taking more than they need. Both run at 30fps rather than 60 — a dot matrix reads identically either way — and the visualiser's ring is laid out once per resize instead of being re-derived, with a thousand-odd `sin`/`cos` pairs and seven freshly grown arrays, on every frame. Since the browser and the model are usually the same machine here, that is budget handed straight back to generation.

If you are short of headroom, `MRT_BITS=4` renders about 10% faster and halves the model's memory, which on an 8GB machine also makes frame times much steadier — measured 34.9ms/frame with a 0.2ms spread, against 8-bit's occasional excursions past 60ms under memory pressure. Its effect on how the music sounds has not been evaluated.

`MRT_BACKEND=mlxfn` is still broken and not recommended. Every graph exported by mlx 0.32.1 and 0.32.2 decodes to white noise — reproducible through the library's own `mrt mlx generate`, at 8-bit and unquantized, with 0, 1 and 2 CFG branches. Google's published `.mlxfn` can't be loaded by those versions either (`[import_function] Invalid string size`), so it appears to need an unreleased mlx. Building mlx from source needs full Xcode for the Metal shader compiler; Command Line Tools alone will not do it.

## Stop:

```bash
./stop.sh
```

This is equivalent to Ctrl+C on the combined launcher and is safe to run more
than once. It stops each complete process group, waits for graceful shutdown,
and force-terminates a group only if it misses the shutdown deadline. PID files
are removed only after no processing remains.
