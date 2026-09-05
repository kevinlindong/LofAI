# Music streaming design

This document records how Magenta RealTime 2 (MRT2) works, how lofAI uses it,
and why the live path is intentionally small. The implementation is pinned to
`magenta-rt[mlx]` 2.0.3; upstream behavior described here was checked against
that release and Google's current model/core documentation.

## MRT2 in one frame

MRT2 is a streaming codec language model, not a prompt-to-song renderer. Every
40 ms it repeats this causal loop:

1. MusicCoCa maps a short text prompt or reference audio to a 768-dimensional
   style embedding and then to 12 style tokens.
2. Optional MIDI supplies the current state of 128 pitches; an optional drum
   token supplies a coarse drum condition.
3. A decoder-only Transformer reads those controls plus its recurrent audio
   history and samples up to 12 SpectroStream audio tokens.
4. SpectroStream decodes the tokens to 1,920 samples of 48 kHz stereo PCM.
5. The new state becomes the context for the next frame.

The small model has 230M parameters. Its local sliding-window attention gives
it an effective context of roughly 20 seconds even though each layer attends to
a shorter window. Controls are frame-aligned, but end-to-end response includes
inference, codec, host, and playback buffering; Google's reference figure is
about 200 ms.

Primary references:

- [MRT2 technical announcement](https://magenta.withgoogle.com/magenta-realtime-2)
- [Official model card](https://github.com/magenta/magenta-realtime/blob/main/MODEL.md)
- [Official repository](https://github.com/magenta/magenta-realtime)
- [Official C++ streaming core](https://github.com/magenta/magenta-realtime/blob/main/core/README.md)

## lofAI's live data path

```text
station / drums / volume
          |
          v
WebSocket control -> Session -> style embedding + recurrent MRT2 state
                                  |
                                  v
                           one MLX worker
                                  |
                         320 ms first PCM burst
                         400 ms steady bursts
                                  |
                                  v
bounded server outbox -> WebSocket -> 8 s AudioWorklet ring
                                      |
                                ~0.6-0.9 s start bank
                                      |
                         loudness gain -> limiter -> speakers
```

There is one loaded model and one MLX-owning worker thread. Each listener keeps
only its own recurrent model state, prompt transition, transport clock, and
seed. The normal live conditioning is:

- six coarse MusicCoCa style tokens from one curated station prompt;
- no piano-roll tokens, so MRT2 performs and continues its own music;
- no drum condition when drums are enabled, or an explicit off token when they
  are disabled;
- the official native runner's sampling defaults: temperature 1.0, top-k 100,
  MusicCoCa CFG 3, notes CFG 5, drums CFG 1.

The older 32-bar symbolic composer remains in `composition.py` and `melody.py`
for offline, fixed-seed listening comparisons. It is not part of `Session` or
the real-time hot loop.

### Component responsibilities

| Component | One job in the live path |
|---|---|
| `frontend/components/music-controls.tsx` | Select a curated station, drums on/off, local volume, or a new take |
| `frontend/lib/mrt-stream.ts` | Own the WebSocket session, choose a small reservoir, and build the output/mastering graph |
| `frontend/public/mrt-pcm-worklet.js` | Consume interleaved PCM through one bounded ring and continuous audio-thread cursor |
| `backend/server.py` | Validate messages and bridge PCM through a bounded per-socket outbox |
| `backend/session_manager.py` | Admit one live listener on this machine, pace chunks, and own the sole model thread |
| `backend/session.py` | Hold one recurrent state/seed and turn station changes into short style ramps |
| `backend/engine.py` | Load MRT2, cache conditioning, render frames, measure speed, and tune codec depth |
| `backend/styles.py` | Define the four listener-facing MusicCoCa prompts and optional audio references |

`composition.py`, `melody.py`, and `backend/evaluation/` are deliberately
outside that table: they support offline listening experiments, not playback.

## What made the previous stream slow

The observed startup calibration was 71.8 ms per 40 ms frame, or 0.56x real
time. A six-second prebuffer can hide that deficit for only about 14 seconds:
the reservoir loses 0.44 seconds every second. Rebuffering was therefore the
expected steady state, not a browser scheduling accident.

Several design choices compounded it:

- Eight-bit eager inference left too little margin on this 8 GB M1 Air.
- One-second first chunks and a two-second server lead made playback and every
  control feel late.
- The browser responded to a sub-real-time producer by banking up to six
  seconds, which postponed rather than solved the next gap.
- Every frame received a generated 128-note piano roll and score clock. That
  added planning and conditioning churn and made the small model sound rigid,
  because the external score competed with its learned audio continuation.
- Nine hidden legacy prompt combinations were embedded during startup even
  though the UI exposed four stations.
- Sampling values differed substantially from the reference live engine,
  especially drum CFG 4 instead of 1.
- Bursty eager inference lets macOS downclock the GPU between calls. Google's
  `RealtimeRunner` explicitly issues tiny GPU operations while its ring is
  full to avoid this.
- The browser and server retained 45 and 30 seconds of audio respectively,
  allowing large amounts of stale music to accumulate without improving
  sustainable throughput.

## Changes and measured tradeoffs

All measurements below were made locally on the target M1 Air with the
`mrt2_small` checkpoint. Real-time factor is seconds of audio generated per
wall-clock second; it must remain above 1.0 indefinitely.

| Eager configuration | ms/frame | real-time factor |
|---|---:|---:|
| 8-bit, 12 codec layers | 37.8 | 1.06x |
| 8-bit, 10 codec layers | 35.4 | 1.13x |
| 8-bit, 8 codec layers | 33.7 | 1.19x |
| 4-bit, 12 codec layers | 35.1 | 1.14x |
| 4-bit, 10 codec layers | 33.1 | 1.21x |

Four-bit weights also reduced model load from roughly 13.4 to 6.1-6.5 seconds
in repeated local runs. The live default is therefore 4-bit, while codec depth
remains adaptive from 12 down to a quality floor of 10. Calibration targets
1.18x, usually selecting 10 or 11 layers on this machine. Listening quality is
subjective, so the evaluation path still supports controlled 8-bit/12-layer
A/B renders when deciding whether that speed-for-detail trade is worthwhile.

Batching still matters for the eager Python runtime:

| Frames per call | Audio duration | observed factor at 4-bit / 10 layers |
|---:|---:|---:|
| 1 | 40 ms | 0.97x |
| 3 | 120 ms | 1.09x |
| 10 | 400 ms | 1.16x |
| 25 | 1,000 ms | 1.20x |

Ten steady-state frames retain enough batching benefit while capping
server-side control granularity at 400 ms. A new take starts with eight frames
so its first 320 ms of audio arrives before the steady batch. The browser starts
after roughly 0.64-0.9 seconds when the renderer is
healthy, keeps playback at exactly 1.0x, and reports pressure early enough for
the backend to reduce one codec layer. It never tries to disguise a sustained
sub-real-time renderer with an ever-growing delay.

An end-to-end WebSocket check of the final defaults produced 10.32 seconds of
PCM in 10.15 seconds of wall time. First PCM arrived in 294 ms, the simulated
browser crossed its playback bank at 616 ms, the largest packet interval was
412 ms, and the reservoir never emptied. The engine held 11 codec layers at
1.20x measured render speed. A separate 30-second run with two station changes
and a drum change produced 30.3 seconds of audio in 30.2 seconds, held roughly
1.17x render speed, and likewise kept the simulated reservoir above zero.

## Why not use the official native runner yet?

Google's production-style macOS examples use the C++ `RealtimeRunner`, an
exported `.mlxfn` graph, a lock-free stereo ring, a dedicated inference thread,
and GPU keepalive. That is the right eventual host architecture.

On the current installation, however, the published graph did not import with
the tested older MLX build, while MLX 0.32.2 imports and runs it but decodes
noise-like output. The graph path measured about 1.12x real time, so enabling it
would improve speed by sacrificing valid audio. Building the official benchmark
also requires the full Xcode Metal toolchain, which is not installed on this
machine. lofAI therefore keeps the verified eager checkpoint path and borrows
the safe runtime ideas: one inference thread, bounded rings, prewarming, and GPU
keepalive.

Revisit native integration when an upstream graph and supported MLX release
round-trip cleanly on this machine. Validate decoded signal and a listening
sample before treating a faster benchmark as usable.

## Operational rules

- `renderRealtimeFactor` in `GET /health` is the primary capacity signal. A
  sustained value below 1.0 is a compute problem, not a buffering problem.
- Keep `MRT_MAX_SESSIONS=1` on an M1 Air. One shared model can hold many states,
  but active listeners divide the same serial inference budget.
- Leave `MRT_CODEBOOKS` unset for live use. Pin 12 only for offline comparison.
- Use `mrt2_base` only offline on this machine; Google specifies the small model
  for Air-class real-time generation.
- Musical changes need blind listening tests. Signal metrics are useful only to
  reject silence, clipping, corruption, or obviously noise-like output.
