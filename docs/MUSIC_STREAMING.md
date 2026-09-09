# Music streaming design

This document records how Magenta RealTime 2 (MRT2) works, how lofAI uses it,
and why the live path is intentionally small. The implementation is pinned to
`magenta-rt[mlx]` 2.0.3; upstream behavior described here was checked against
that release and Google's current model/core documentation.

## MRT2 in one frame

MRT2 is a streaming codec language model, not a prompt-to-song renderer. Every
40 ms it repeats this causal loop:

1. MusicCoCa supplies 12 style tokens from a 768-dimensional text/audio
   embedding. lofAI caches this work until the style changes.
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
                         fixed makeup -> limiter -> speakers
```

There is one loaded model and one MLX-owning worker thread. Each listener keeps
only its own recurrent model state, prompt transition, transport clock, and
seed. The normal live conditioning is:

- all twelve MusicCoCa style tokens from one curated station prompt. Neither
  checkpoint was trained with style-token masking (`mask_musiccoca=False`),
  so masking the fine half at inference conditions the model outside its
  training distribution; the earlier six-token mask audibly weakened style
  adherence and is now an explicit experiment behind
  `MRT_STYLE_TOKEN_LEVELS`;
- no piano-roll tokens, so MRT2 performs and continues its own music;
- no drum condition when drums are enabled, or an explicit off token when they
  are disabled;
- sampling: temperature 1.0, top-k 100, MusicCoCa CFG 3, drums CFG 1, and
  notes CFG 1 (the library default). The earlier notes CFG of 5 told the
  model, every frame, to strongly follow a fully masked score - a token
  combination it never saw in training, since the notes CFG token co-varied
  with real piano-roll data. Raise it only when actually supplying notes.

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
| `backend/take_health.py` | Watch each take's quiet-moment floor and detect a self-amplifying hiss bed; provide the crossfade splice |
| `backend/engine.py` | Load MRT2, cache conditioning, render frames, measure speed, and tune codec depth |
| `backend/fast_engine.py` | Specialize the pinned magenta-rt streaming step: sliced logits, cached conditioning encoding, hoisted constants, clean RVQ truncation |
| `backend/compiled_engine.py` | Render a chunk as tokens first, then one batched codec call; trace both halves with `mx.compile`; keep session state detached from chunk activations |
| `backend/style_tokens.py` | NumPy replica of MusicCoCa's RVQ tokenizer, verified against TFLite, so the interpreters can be released after startup |
| `backend/styles.py` | Define the four listener-facing MusicCoCa prompts and optional audio references |

`composition.py`, `melody.py`, and `backend/evaluation/` are deliberately
outside that table: they support offline listening experiments, not playback.

## What made the previous stream slow

### September 2026: where a frame actually went

A stage-by-stage measurement of one 12-codebook frame on an M3 Pro (18 GB,
8-bit weights, barrier after each stage) corrected the earlier picture:

| Stage | ms | Note |
|---|---:|---|
| Conditioning encoder | 0.2 | cached per block |
| Temporal body (12 layers, 1024-d) | 4.9 | about 200 MB of weight reads |
| Depth loop (12 sequential steps) | 6.6 | 14.8 with a barrier after every step, which is how the earlier profile attributed "14 of 24 ms" here |
| SpectroStream codec, one frame | 10.3 | 143 MB of float32 Conv2D plus the iSTFT, run at T = 1 |
| Python graph construction | 5.6 | on the critical path; `async_eval` hides only part of it |

Two findings drove the changes. First, `nn.quantize` never touches the
codec: it converts `nn.Linear`/`nn.Embedding` and the attention modules'
own `to_quantized` hooks, so the 143 MB conv decoder and the 67 MB RVQ table
stay float32 and are read in full every 40 ms. Per-frame weight traffic is
roughly 540 MB, which on an M1's ~68 GB/s is an ~8 ms floor before any
kernel launch or Python overhead. Second, the codec - not the depth loop -
was the largest single stage.

**Batched codec.** The codec is causal convolution, not autoregression: its
`step` over a ten-frame token sequence yields the same samples as ten
single-frame steps (measured: 1 LSB over 600k samples), at 2.9 instead of
10.0 ms per frame. `compiled_engine.ChunkRenderer` therefore samples every
frame's tokens first and decodes the chunk in one call. Latency is
unchanged: `generate()` already returned whole chunks. The one trap was
memory: the codec's returned streaming state consists of slices into the
chunk's activations, and MLX slices share their donor buffer, so a naive
version pinned about a gigabyte per listener between chunks. State leaves
are now copied out with `mx.contiguous` before they leave the renderer, and
the flatten/rebuild helpers avoid self-referential closures, which had been
holding those same views hostage to the cyclic garbage collector.

**Compiled step.** The sampling half of the specialized step (the encoder
runs once per block outside the trace) and the batched codec are traced
with `mx.compile`, with sequence_layers' `Sequence`-bearing state flattened
to arrays at the boundary. Python time per depthformer frame fell from 4.3
to 0.55 ms and the step from 12.5 to 7.2 ms; fused kernels, not just less
Python. The active codebook count is Python control flow inside the step, so
one trace is kept per count, and every live chunk length is traced during
calibration (`prewarm`, about 0.6 s for three counts at two lengths).
Compiled output is not bit-identical: from an identical state, compiled and
eager tokens differed in 10 of 60 single steps, all in the flat-distribution
deep codebooks, with a total variation distance of about 0.03 on the
sampling distribution and the same argmax and top-5. The already-shipped
sliced-logits step differs from the stock library in 30 of 30 frames, so
this is the same class of trade. `MRT_COMPILE=0` restores the eager step.

Combined, on the M3 Pro at 12 codebooks: 18.8 → 8.5 ms per frame (2.2x),
with the codebook dial now worth about 0.3 ms per step. The M1 Air was not
available for measurement; both changes attack costs that scale with memory
bandwidth and CPU speed, which the M1 has less of, so the relative gain
should be at least as large there. Casting the codec to fp16/bf16 was also
tried and dropped: 0.7 ms per frame at best, and the conv decoder's output
did not survive the cast cleanly.

**Memory.** The MusicCoCa TFLite interpreters (text encoder, mapper,
quantizer) stayed resident for the life of the process - about 900 MB on a
cold start - and JAX is imported by the vendored `sequence_layers.mlx`
(~230 MB). The RVQ quantizer graph turned out to be a plain 12-level
residual nearest-neighbour search; `style_tokens.py` extracts its codebooks,
verifies its own tokens against the interpreter on random vectors, the
station embeddings, and their ramp blends (956 of 956 agreed locally), and
caches them next to the embeddings. With tokenization native, the engine
releases every interpreter after warm-up; they rebuild lazily if an unknown
prompt arrives. The MLX buffer cache limit is applied from the first
allocation rather than after calibration, which removed a 1.5-3 GB startup
transient. Cold start settled at about 1.0 GB resident versus 1.7 GB before;
the JAX import remains, since `sequence_layers.mlx` uses it for its config
base classes.

### September 2026 audit of the installed runtime

The optimized step was configured on but failed during installation:
`inspect.signature(mx.quantized_matmul)` raises `TypeError` on MLX 0.32.2's
native nanobind function. The exception sent generation back to the stock
step, disabling sliced projections, cached conditioning, and hoisted constants.
The projector now reads the quantization mode from `QuantizedLinear`, matching
the installed layer's own call. This enables the existing optimization on the
actual pinned runtime rather than relying on its configuration flag.

SpectroStream also rebuilt its constant inverse-STFT synthesis window with
GPU operations and copied it to NumPy inside every frame. That readback
synchronized the pipeline. Caching the original window once removes the
repeated calculation and synchronization without changing its values.

Two forward/reverse-order local comparisons on the 8 GB M1 used the small
model, 8-bit weights, all 12 codebooks, 10-frame paced calls, GPU keepalive,
and the 384 MB MLX cache cap. Each variant rendered 150 frames, excluding
30 warmup frames from timing:

| Configuration | Mean ms/frame, run 1 | Mean ms/frame, run 2 |
|---|---:|---:|
| Previous deployed fallback | 41.65 | 39.02 |
| Enabled specialized step | 36.84 | 36.38 |
| Specialized step + cached window | 36.20 | 36.29 |

The combined average fell from 40.33 to 36.25 ms/frame (about 10% less render
time, approximately 0.99x to 1.10x real time at full depth). The cached-window
and uncached-specialized paths produced identical six-second seeded PCM in
both comparisons. MLX active allocations were about 450 MB and reusable cache
about 239 MB; these are MLX measurements, not total application memory or
energy use. Host load and swap pressure affected timings. Adaptive 10–12
codebooks remain enabled to obtain additional margin when full depth cannot
meet the 1.18x target. An experimental compiled sampler was not retained:
its smaller additional improvement was inconsistent.

The old speed meter also used the median chunk cost. With two 30 ms frames
and one 200 ms frame it reported 1.33x, even though total throughput was only
0.46x. The meter now divides total generated audio by total rendering time,
weighted correctly for unequal chunk sizes. The tuner can ignore one worst
chunk to preserve quality through isolated jitter, but a deficit that remains
for six seconds still causes a depth reduction. Admission and browser reservoir
selection use the untrimmed measured throughput.

An integrated 180-second source-audio capture then reproduced six simulated
playback underruns on this host. Live inference RTF ranged from 0.705 to 1.188
(median 1.108), despite the improved mean benchmark. Unpaused packet intervals
reached 1.025 seconds. This exposed a separate client bug: its 0.05 RTF update
deadband ignored a 1.203-to-1.157 change even though it crossed the 1.18 buffer
policy boundary. Valid speed updates now always refresh the policy, and the
server broadcasts speed every four seconds even when depth stays at its floor.
Each audible underrun adds 0.2 seconds of recovery margin, capped at 0.8 extra
seconds; the margin survives pauses and resets for a new take/session.

Replaying the same packet arrivals with recorded periodic health measurements
and the new bounded margin reduced interruptions from six to three. Total
refill waits were 5.92 seconds versus 5.32 seconds: fewer, longer recoveries.
Initial startup stayed unchanged and peak retained audio was 2.013 seconds.
This is a fixed-trace simulation, not a second live performance or a guarantee
of uninterrupted audio. A slower initial start did not reduce interruptions
on that trace, so the initial reservoir defaults remain unchanged. No finite
reservoir fixes sustained generation below 1x at the configured quality floor.

A subsequent live 90-second capture verified periodic status, a station change,
pause/resume, and the new margin cap. With heavier concurrent host activity,
median reported render RTF was 0.975 at the 10-layer floor, unpaused packet
intervals reached 1.637 seconds, and five simulated underruns required 13.57
seconds of refill waits. Peak buffering remained bounded at 2.413 seconds.
The server shut down cleanly after both captures. The new buffer policy cannot
promise uninterrupted playback when the host cannot supply frames fast enough;
the two live captures are not controlled before/after speed comparisons.

Those checks exposed a scheduling mismatch: tool-launched Python's main
thread had macOS user-interactive QoS, while a newly created Python worker
had default QoS. A final paired comparison ran on a dedicated Python worker
at 8-bit / 10 codebooks, alternating default and user-initiated QoS:

| Worker priority | Forward mean ms/frame | Reverse mean ms/frame |
|---|---:|---:|
| Default | 39.88 | 52.63 |
| User initiated | 37.97 | 39.85 |

All four seeded 4.8-second outputs were identical. User-initiated was faster
in both pairs, but the large final default-thread stall makes the aggregate
15.9% improvement noisy. This effect must not be added to the earlier 10%
figure or treated as an assurance of real-time performance under every load.
The dedicated inference worker now requests user-initiated QoS before loading
MLX, using Apple's supported per-thread API. Higher existing priority is
preserved, failure falls back safely, and no other app's priority is changed.
`MRT_WORKER_QOS=default` opts out; `/health` exposes `workerQoS`. This follows
[Apple's guidance for work needed for an immediate user action](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/EnergyGuide-iOS/PrioritizeWorkWithQoS.html).
The long live captures above preceded this scheduling change; the paired
worker benchmark and regression tests validate it separately.

The observations and tables below predate this audit and describe earlier
configurations, not guaranteed throughput under the current machine load.

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
in repeated local runs. Four-bit was originally chosen as the live default on
those numbers alone. It was later reverted to 8-bit: `nn.quantize` was
quantizing the *entire* sampler with round-to-nearest 4-bit weights -
including the SpectroStream codec decoder that turns tokens into waveforms
and every embedding table - and that was audibly the largest quality cost in
the pipeline. The specialized step loop below buys back more time than the
4-bit/8-bit gap, so the speed argument for 4-bit no longer holds. When
`MRT_BITS=4` is explicitly requested for memory, the codec decoder now stays
at 8-bit: degraded token prediction is a taste choice, a degraded codec is
just noise. Codec depth remains adaptive from 12 down to a quality floor of
10, and calibration targets 1.18x.

## The specialized streaming step

Profiling the per-frame step on an M3 Pro (barriered, so relative numbers
only) attributed roughly 14 of 24 ms to the depth loop: 10-12 sequential
two-layer transformer steps, each ending in a `to_logits` projection of its
768-dim hidden state onto the full 12,294-token vocabulary, of which exactly
1,024 logits are valid for that codebook. `backend/fast_engine.py` replaces
the pinned library step (per instance, guarded by version and structure
checks, with per-call fallback to stock) with one that:

- projects each depth step against only its codebook's 1,024 weight rows,
  reading 12x less `to_logits` weight data per step. Metal tiles the smaller
  matmul differently, so logits may differ from stock in the final bf16
  mantissa bit; install-time verification requires agreement within one ulp
  and disables slicing otherwise. Takes can therefore diverge from the stock
  trajectory over time, the same documented trade the fast sampler already
  makes with its RNG stream;
- computes the conditioning encoder once per conditioning block instead of
  every frame (the encoder is stateless in this configuration, so this is
  exactly equal);
- hoists the depth transformer's initial state, the skipped-codebook dummy
  tokens, and the CFG/delay bookkeeping the live path never uses out of the
  per-frame loop (exactly equal).

It also fixes what quality truncation actually decodes. Upstream pads
skipped codebooks with code 0 and the RVQ decode then adds that codebook's
row-0 *centroid* - a full-magnitude learned vector, not silence - into every
frame (measured `|q10[0]| = 6.24` against a codebook mean of `6.23`). At 10
active layers that contaminated every frame of audio with two arbitrary
residual vectors, which is a large part of why reduced depth sounded broken
rather than merely duller. The decode now slices the token frame to the
active count, making truncation mean truncation. At 12 active layers both
paths are identical.

Measured on an M3 Pro at 12 codec layers, 10-frame chunks, in the pipelined
engine loop (medians over 250 frames): 8-bit stock+fast-sampler 20.1
ms/frame (1.99x) against 19.0 ms/frame (2.11x) with the fast engine, and the
same take stayed bit-identical end to end. At 10 active layers the gap was
19.6 against 18.0 ms/frame, where output intentionally differs because the
code-0 contamination is gone. The absolute win is larger on
bandwidth-constrained machines like the target M1 Air, where the full
`to_logits` read alone costs roughly 1.9 ms per frame at 8 bits versus
roughly 0.16 ms sliced.

## Long takes: the rising noise floor

Two separate mechanisms made long sessions grow an audible hiss bed, one on
each side of the WebSocket, and they compounded.

**The model amplifies its own floor.** MRT2's only continuity is the audio
it just generated. On sparse stations that feedback has a failure
attractor: once a bright sustained texture enters the ~20-second context,
the model tends to continue and reinforce it. Rendered 12-minute takes,
measured on the level of the quietest 50 ms blocks (the gaps between notes,
where a bed is exposed) against the same take's first minute:

| Station | quiet-gap floor early | worst trailing minute | high band (5-14 kHz) |
|---|---:|---:|---:|
| rainy-piano | -42.7 dBFS | +9.9 dB | +28 dB |
| dusty-beats (12 min) | -45.3 dBFS | +19.4 dB | +23 dB |
| jazz-cafe | -37.8 dBFS | +6.6 dB, receded | +12 dB, receded |

The first two are runaways: the floor climbs for minutes and does not come
back. The jazz excursion is what a healthy arrangement getting brighter for
half a minute looks like, and it recovers on its own.

`backend/take_health.py` watches exactly this measurement on the PCM each
listener actually receives. A baseline floor profile is frozen over the
take's first minute (after a short landing period); the high band (5-14
kHz) of the trailing minute's floor must then rise at least 10 dB over its
own baseline and clear an absolute audibility gate, in at least 80% of
evaluations across a 40-second audio window, before the take is declared
drifted. The decision is deliberately spectral: a live-captured failure
grew +22 dB of high-band hiss while its overall floor rose only +1 dB -
the bed brightens long before it lifts - while a warm rumble or denser
bass never qualifies. Level and spectrum are measured on the same
quietest-decile blocks, so louder or denser playing does not qualify
either - only the bed under it. The repair is a server-side equal-power
crossfade onto a fresh recurrent state under the same conditioning (one
extra chunk of render cost, deterministic refresh seed, no transport or
session change): a subtle track change instead of a slowly degrading
stream. Replayed against the captured takes, the guard fires mid-runaway
on both failures, twice on the worst, never on the healthy 6-minute take,
and costs one borderline cymbal-wash passage a single crossfade. Station
changes re-learn the baseline, since the recurrent state - and any drift
it carries - survives them. `MRT_TAKE_GUARD=0` disables the guard.

The September audit found that folding PCM to mono before measuring it hid
opposite-phase stereo hiss. The monitor now averages channel powers instead,
and evaluates every 400 ms of audio rather than once per incoming chunk.
Chunk size and empty calls therefore cannot shorten or lengthen the sustain
window. Small remainder buffers own their memory, and percentile selection
uses partitioning instead of sorting. Synthetic tests reproduce both bugs;
monitor overhead remains about 0.34 ms per 400 ms chunk on the local M1.
The guard needs an early baseline: it does not remove hiss already present
from the beginning or guarantee detection after a noisy style-change baseline.
The affected station prompts no longer explicitly ask for vinyl or tape.

**The client mastering chased dynamics.** The old loudness normalizer
adapted over a +9 dB range fast enough to follow musical passages: every
mellow stretch ratcheted the gain up - exactly when a floor is most
audible - and the codec noise floor rose with it, up to +6.9 dB in
simulation over a real take. A later ±3 dB trim still reached its maximum
in about 15 seconds on quiet audio. The current graph removes that adaptive
trim, its extra 32,768-sample analyser and its polling timer altogether.
A fixed +5 dB makeup stage compensates for most of the codec's -6 dB int16
headroom; the existing limiter, output ceiling, and listener volume remain.
Music and any inherent noise are amplified equally by that fixed amount,
without a gain increase as the take becomes quiet.

The worklet now retains incoming new-take PCM while the previous take fades
out, honors a pause during that transition, and fades at a ring overflow
instead of silently omitting a packet and splicing later audio across the hole.
At 48 kHz and unity playback speed, it reads each PCM sample directly instead
of interpolating with a second ring read. Resampling retains one continuous
fractional cursor when source and device rates differ.
In a synthetic native-rate worklet benchmark, 80 seconds of stereo audio took
28.4 ms of processing versus 314.6 ms before (five-run medians). This is the
isolated DSP loop, not an estimate of overall app CPU or model speed. The
timer-based compatibility sink also preserves its queued scheduling cursor
across a pause so new chunks cannot overlap already scheduled audio.

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
