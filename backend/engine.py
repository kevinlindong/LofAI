# magenta realtime 2 inference engine

import logging
import os
import hashlib
import importlib.metadata
from pathlib import Path
import tempfile
import threading
import time
from collections import deque

import numpy as np

from audio_quality import require_startup_pcm_quality
from melody import piano_roll

log = logging.getLogger(__name__)


class EngineStopping(RuntimeError):
    """Raised on the model thread when application shutdown is requested."""

# mrt2 emits 40ms frames of 48khz stereo audio
FRAMES_PER_SECOND = 25
SAMPLE_RATE = 48000
CHANNELS = 2
FRAME_SECONDS = 1.0 / FRAMES_PER_SECOND

# spectrostream stacks residual quantisers per frame and the depthformer samples
# them one after another, so the codebook count is very nearly a dial on how
# long a frame takes to render: about 1.1ms of a 40ms budget each on an m1.
#
# Eight layers still decodes, but same-seed listening diagnostics found a much
# darker, bass-heavier balance there than at full depth. Skipped tokens also
# enter the next temporal state, so this is not a free high-frequency-only
# trade. Listener-facing generation therefore keeps at least ten by default
# and always restores the highest count the machine can sustain.
MAX_CODEBOOKS = 12
# Eight layers keeps gross codec metrics intact, but listening diagnostics show
# that it materially darkens the mix and the dummy tail also feeds the next
# recurrent state. Ten is the default floor for listener-facing audio. An
# explicit environment override can still lower it for controlled experiments.
MIN_CODEBOOKS = 10
ABSOLUTE_MIN_CODEBOOKS = 8

# Frames per calibration probe. One global burn-in fills kernels, then each
# candidate is measured on the same one-second transport granularity listeners
# receive. Long enough that one slow frame does not decide the answer, while a
# full walk to the configured floor still adds only a few seconds to startup.
PROBE_FRAMES = 25

# how long the tuner sits still after a change. long enough that the estimate
# has actually caught up with the new setting, so it cannot chase itself.
TUNE_DWELL_SECONDS = 6.0

# how many recent chunks the render-speed estimate is taken over, and - the
# important part - taken as a median rather than a mean. on a machine under
# memory pressure a single chunk can take twice as long as its neighbours, and
# an average lets that one chunk spend a codebook of everybody's audio. a
# median asks the machine to actually be slow, not to have hiccupped once.
COST_WINDOW = 9
MIN_TUNE_SAMPLES = 5


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _mlx_version() -> str:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return "?"


class MRTEngine:
    # wraps one loaded mrt2 model, shared by every session
    #
    # the model itself is stateless across calls: generate() takes the
    # streaming state in and hands a new one back, so a single loaded model can
    # serve any number of sessions as long as each keeps its own state.
    #
    # IMPORTANT: load() and generate() must run on the same thread, for the
    # whole life of the process. mlx streams are thread local, so a model loaded
    # on one thread and called from another dies with "There is no Stream(gpu,
    # 1) in current thread". SessionManager satisfies this by loading and
    # generating on its single worker thread. the musiccoca tflite interpreters
    # behind embed() are not thread safe either, so they want the same thread.

    def __init__(self):
        self.size = os.environ.get("MRT_MODEL_SIZE", "mrt2_small")
        # "python" builds the model from the safetensors checkpoint and runs it
        # eagerly. "mlxfn" runs a graph exported by `mrt mlx export`, which is
        # nominally faster - but every graph exported by mlx 0.32.x (the newest
        # on pypi) decodes to white noise: no energy below 200hz, 91% above
        # 4khz, zero crossing rate 0.50. that reproduces through the library's
        # own `mrt mlx generate` CLI, at 8-bit and unquantized, and with 0, 1
        # and 2 cfg branches, so it is the exporter and not this app. the same
        # version gap stops mlx 0.32.x importing google's published .mlxfn at
        # all. keep the eager path until a newer mlx ships; whatever the export
        # was worth, the eager path pipelines now and has closed some of it.
        self.backend = os.environ.get("MRT_BACKEND", "python")
        self.bits = _env_int("MRT_BITS", 8)
        # Match the stable sampling centre of Magenta's listener-facing MRT2
        # runtime. The melody guide below supplies structure, so randomness can
        # stay in the performance instead of deciding whether a tune exists.
        self.temperature = _env_float("MRT_TEMPERATURE", 1.0)
        self.top_k = _env_int("MRT_TOP_K", 100)
        self.cfg_musiccoca = _env_float("MRT_CFG_MUSICCOCA", 3.0)
        self.cfg_notes = _env_float("MRT_CFG_NOTES", 5.0)
        self.cfg_drums = _env_float("MRT_CFG_DRUMS", 1.0)

        # The official live engine keeps the coarse half of MusicCoCa's RVQ
        # tokens and masks its fine tail. Broad musical style survives while
        # brittle prompt-specific detail does not oversteer every frame.
        self.style_token_levels = max(
            1, min(12, _env_int("MRT_STYLE_TOKEN_LEVELS", 6))
        )

        # what the auto-tuner aims for. anything under 1.0 means the machine
        # renders slower than it plays, which no amount of buffering can hide
        # for long; the margin over 1.0 covers the model being slower on some
        # passages than others and the rest of the box wanting the gpu too.
        #
        # this is deliberately modest. codebooks are the currency being spent
        # and they are audible, so the aim is enough headroom to ride out a
        # wobble - not enough to never think about it again. gaps the listener
        # actually reports buy an extra step down, which is the honest signal.
        self.target_rtf = _env_float("MRT_TARGET_RTF", 1.15)
        self.mlx_cache_mb = _env_int("MRT_MLX_CACHE_MB", 384)
        self.fast_sampler_enabled = _env_int("MRT_FAST_SAMPLER", 1) != 0

        # 0 keeps the auto-tuner; any other value pins the codebook count
        self.pinned_codebooks = _env_int("MRT_CODEBOOKS", 0)
        self.min_codebooks = max(
            ABSOLUTE_MIN_CODEBOOKS,
            min(MAX_CODEBOOKS, _env_int("MRT_MIN_CODEBOOKS", MIN_CODEBOOKS)),
        )
        self.codebooks = MAX_CODEBOOKS
        self.max_codebooks = MAX_CODEBOOKS

        self._system = None
        self._warm = False
        self._style_key = None
        self._notes_key = None
        self._embeddings: dict[str, np.ndarray] = {}
        self._blocks: dict[tuple[str, int | None], tuple] = {}
        self._load_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self.load_error: str | None = None

        # fast path handles, filled in by _prepare_fast_path
        self._fast = False
        self._fast_sampling = False
        self._sampler = None
        self._input_spec = None
        self._depth_config = None

        # recent render cost in seconds per frame, for the auto-tuner
        self._costs: deque[float] = deque(maxlen=COST_WINDOW)
        self._cost_lock = threading.Lock()
        self._last_tune = 0.0
        cache_root = os.environ.get(
            "MRT_EMBEDDING_CACHE",
            str(Path.home() / "Library" / "Caches" / "lofai" / "embeddings"),
        )
        self._embedding_cache_dir = Path(cache_root).expanduser() if cache_root else None

    @property
    def ready(self) -> bool:
        # not just loaded but warmed: embedding the prompts takes a few seconds,
        # and a session promoted before that finishes would report itself live
        # while the listener sits in silence
        return self._system is not None and self._warm

    def realtime_factor(self) -> float:
        # seconds of audio rendered per second of wall clock, as measured
        cost = self._typical_cost()
        if cost <= 0.0:
            return 0.0
        return FRAME_SECONDS / cost

    def _typical_cost(self) -> float:
        with self._cost_lock:
            costs = tuple(self._costs)
        if not costs:
            return 0.0
        ordered = sorted(costs)
        return ordered[len(ordered) // 2]

    def _seed_cost(self, cost: float):
        # start the window off at what calibration just measured, so the first
        # listener is judged against a real number rather than one chunk
        with self._cost_lock:
            self._costs.clear()
            for _ in range(COST_WINDOW):
                self._costs.append(cost)

    def _clear_costs(self):
        with self._cost_lock:
            self._costs.clear()

    def prepare_start(self):
        self._stop_requested.clear()
        self.load_error = None

    def request_stop(self):
        self._stop_requested.set()

    def _raise_if_stopping(self):
        if self._stop_requested.is_set():
            raise EngineStopping("engine shutdown requested")

    def load(self):
        # build the model and warm it up - takes tens of seconds
        with self._load_lock:
            self._raise_if_stopping()
            if self._system is not None:
                return

            from magenta_rt.config import MUSICCOCA, PIANOROLL_WITH_ONSETS

            self._style_key = MUSICCOCA.key
            self._notes_key = PIANOROLL_WITH_ONSETS.key
            started = time.monotonic()

            if self.backend == "mlxfn":
                log.warning(
                    "MRT_BACKEND=mlxfn: exported graphs decode to noise under "
                    "mlx %s. only set this once mlx can export this model again.",
                    _mlx_version(),
                )
                try:
                    self._system = self._load_mlxfn()
                except Exception as exc:  # noqa: BLE001
                    # the published .mlxfn needs a newer mlx than pip ships
                    log.warning(
                        "could not load the exported graph (%s), using the "
                        "python model instead",
                        exc,
                    )
                    self.backend = "python"

            if self._system is None:
                self._system = self._load_python()
                self._raise_if_stopping()
                self._prepare_fast_path()

            log.info(
                "model loaded in %.1fs (%s backend)",
                time.monotonic() - started,
                self.backend,
            )

    def _load_mlxfn(self):
        # exported graph, if a future mlx ever exports one that decodes
        from magenta_rt.mlx.system import MagentaRT2SystemStdMlxfn

        log.info("loading %s (mlxfn backend)", self.size)
        return MagentaRT2SystemStdMlxfn(
            size=self.size,
            temperature=self.temperature,
            top_k=self.top_k,
            cfg_scales={
                "musiccoca": self.cfg_musiccoca,
                "notes": self.cfg_notes,
                "drums": self.cfg_drums,
            },
        )

    def _load_python(self):
        # model built and quantized at load time from the safetensors checkpoint
        from magenta_rt.mlx.system import MagentaRT2System

        log.info("loading %s (python backend, %d-bit)", self.size, self.bits)
        return MagentaRT2System(
            size=self.size,
            bits=self.bits,
            temperature=self.temperature,
            top_k=self.top_k,
            cfg_scales={
                "musiccoca": self.cfg_musiccoca,
                "notes": self.cfg_notes,
                "drums": self.cfg_drums,
            },
        )

    def _prepare_fast_path(self):
        # our own step loop needs three things the library keeps private: the
        # sampler, its conditioning builder, and the depthformer config that
        # carries the codebook count. if a future release moves any of them we
        # fall back to the stock generate() - slower and with a fixed quality,
        # but working, which beats refusing to start.
        try:
            import mlx.core as mx
            import sequence_layers.mlx as sl

            # held rather than imported per chunk: sequence_layers is vendored
            # behind an import hook, and the generate loop should not be paying
            # to rediscover it
            self._mx = mx
            self._sl = sl

            if self.fast_sampler_enabled:
                try:
                    from magenta_rt.mlx import depthformer
                    from fast_sampler import install as install_fast_sampler

                    self._fast_sampling = install_fast_sampler(
                        depthformer, mx, self.top_k
                    )
                except Exception as exc:  # noqa: BLE001 - optional specialization
                    log.warning("fast sampler unavailable; using Magenta's: %s", exc)
                    self._fast_sampling = False

            system = self._system
            self._sampler = system._sampler
            self._input_spec = sl.ChannelSpec(
                shape=(system._num_channels,), dtype=mx.int32
            )
            system._build_conditioning({}, None, None, None)

            config = system._sampler.depthformer.sampler.decoder.config
            if hasattr(config, "num_active_codebooks"):
                self._depth_config = config
                self.max_codebooks = int(config.num_codebooks)
                self.codebooks = self.max_codebooks
            self._fast = True
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "the library internals this build pipelines against have moved "
                "(%s); falling back to the stock generate loop",
                exc,
            )
            self._fast = False

    # --- style ---

    def embed(self, prompt: str) -> np.ndarray:
        # style embedding for a prompt, cached (text encoding is a tflite call)
        cached = self._embeddings.get(prompt)
        if cached is not None:
            return cached

        # The mapper projects text into the audio side of MusicCoCa's shared
        # space before RVQ. This is the path used by Magenta's own MLX CLI and
        # native runtime; without it most prompt tokens differ and conditioning
        # is markedly less faithful. Embeddings are warmed once, so it adds no
        # cost to live generation.
        embedding = np.asarray(
            self._system.embed_style(prompt, use_mapper=True), dtype=np.float32
        )
        self._embeddings[prompt] = embedding
        return embedding

    def warm_embeddings(self, prompts: list[str]):
        # pre-embed every prompt so style changes never wait on the text encoder
        self._load_embedding_cache(prompts)
        for prompt in prompts:
            self._raise_if_stopping()
            self.embed(prompt)
        self._save_embedding_cache(prompts)

        # MusicCoCa builds its RVQ interpreter lazily. Build and cache every
        # fixed conditioning block now so no listener pays that startup cost.
        if prompts:
            for prompt in prompts:
                self._raise_if_stopping()
                self._conditioning(self._embeddings[prompt], prompt)

    def _embedding_cache_path(self, prompts: list[str]) -> Path | None:
        if self._embedding_cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(b"lofai-musiccoca-mapped-v1\0")
        for prompt in prompts:
            digest.update(prompt.encode("utf-8"))
            digest.update(b"\0")
        resource_dir = getattr(self._system._style_model, "_resource_dir", None)
        if resource_dir is not None:
            for name in ("text_encoder.tflite", "mapper.tflite"):
                path = Path(resource_dir) / name
                try:
                    stat = path.stat()
                    digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
                except OSError:
                    return None
        return self._embedding_cache_dir / f"{digest.hexdigest()}.npz"

    def _load_embedding_cache(self, prompts: list[str]):
        path = self._embedding_cache_path(prompts)
        if path is None or not path.is_file():
            return
        try:
            with np.load(path, allow_pickle=False) as cached:
                stored_prompts = cached["prompts"].tolist()
                embeddings = cached["embeddings"]
            if stored_prompts != prompts or embeddings.shape != (len(prompts), 768):
                return
            if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
                return
            self._embeddings.update(zip(prompts, embeddings, strict=True))
            log.info("loaded %d mapped style embeddings from cache", len(prompts))
        except (OSError, ValueError, KeyError):
            log.warning("ignoring invalid style embedding cache %s", path)

    def _save_embedding_cache(self, prompts: list[str]):
        path = self._embedding_cache_path(prompts)
        if path is None or not prompts or not all(p in self._embeddings for p in prompts):
            return
        if path.is_file():
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            embeddings = np.stack([self._embeddings[p] for p in prompts]).astype(np.float32)
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".npz", dir=path.parent, delete=False
            ) as temp:
                temp_path = Path(temp.name)
                np.savez_compressed(temp, prompts=np.asarray(prompts), embeddings=embeddings)
            os.replace(temp_path, path)
            log.info("cached %d mapped style embeddings", len(prompts))
        except OSError as exc:
            log.warning("could not cache style embeddings: %s", exc)

    def _conditioning(
        self, style: np.ndarray, key: str | None, note: int | None = None
    ):
        # Tokenize style plus the current melodic guide note into the block the
        # sampler conditions on. `key` caches fixed prompt/note combinations;
        # blended embeddings during a slider ramp are each seen only once.
        cache_key = (key, note) if key is not None else None
        if key is not None:
            cached = self._blocks.get(cache_key)
            if cached is not None:
                return cached

        tokens = list(self._system._style_model.tokenize(style))
        tokens[self.style_token_levels :] = [-1] * (
            len(tokens) - self.style_token_levels
        )
        conditioning = {self._style_key: tokens}
        notes = piano_roll(note)
        if notes is not None:
            conditioning[self._notes_key] = notes
        built = self._system._build_conditioning(
            conditioning, None, None, None
        )
        if cache_key is not None:
            self._blocks[cache_key] = built
        return built

    # --- quality dial ---

    def set_codebooks(self, count: int) -> bool:
        # how many of spectrostream's residual quantisers the depthformer
        # actually samples. the rest are filled with a dummy token, so shapes -
        # and therefore a live session's state - are unaffected.
        if self._depth_config is None:
            return False

        count = max(self.min_codebooks, min(self.max_codebooks, int(count)))
        if count == self.codebooks:
            return False

        # the config is a frozen dataclass; this is the only field we touch
        object.__setattr__(self._depth_config, "num_active_codebooks", count)
        self.codebooks = count
        return True

    def calibrate(self):
        # pick a codebook count this machine can actually sustain, rather than
        # making the first listener sit through the runtime tuner walking down
        # to it one step at a time.
        #
        # this is the last thing that happens before the engine reports itself
        # ready, so nobody is promoted into a stream mid-measurement.
        self._raise_if_stopping()
        self._calibrate()
        self._raise_if_stopping()
        self._trim_mlx_cache()
        self._warm = True

    def _trim_mlx_cache(self):
        # Loading, quantizing and probing leaves roughly 1.5GB of reusable MLX
        # buffers on an 8GB M1, while steady streaming needs about 240MB. Give
        # that headroom back to the OS to avoid swap-driven frame spikes.
        if self.mlx_cache_mb <= 0:
            return
        try:
            mx = self._mx if self._fast else __import__("mlx.core", fromlist=["core"])
        except (ImportError, AttributeError):
            return
        before = mx.get_cache_memory()
        mx.set_cache_limit(self.mlx_cache_mb * 1024 * 1024)
        mx.clear_cache()
        after = mx.get_cache_memory()
        log.info(
            "trimmed MLX cache %.0fMB -> %.0fMB (limit %dMB)",
            before / (1024 * 1024),
            after / (1024 * 1024),
            self.mlx_cache_mb,
        )

    def _calibrate(self):
        # Every backend must prove that it decodes plausible audio before ready
        # becomes true. This catches silence, clipping, and the known mlxfn
        # white-noise failure before a listener can ever receive its PCM.
        prompt, style = next(iter(self._embeddings.items()))
        plan = ((style, prompt, None, PROBE_FRAMES),)
        quality_pcm = []
        pcm, probe_state = self.generate(None, plan)
        quality_pcm.append(pcm)

        if not self._fast or self._depth_config is None:
            started = time.monotonic()
            pcm, _ = self.generate(probe_state, plan)
            cost = (time.monotonic() - started) / PROBE_FRAMES
            quality_pcm.append(pcm)
            self._seed_cost(cost)
            self._require_audio_quality(quality_pcm)
            return

        # One global burn-in is enough: changing the active count changes loop
        # length, not tensor/kernel shapes. Probe at the real 25-frame chunk
        # size so pipeline-drain overhead is represented accurately.

        if self.pinned_codebooks:
            self.set_codebooks(self.pinned_codebooks)
            cost, _, pcm = self._probe(self.codebooks, probe_state, plan)
            quality_pcm.append(pcm)
            self._seed_cost(cost)
            self._require_audio_quality(quality_pcm)
            log.info(
                "codebooks pinned to %d (%.1f ms/frame, %.2fx real time)",
                self.codebooks,
                self._typical_cost() * 1000,
                self.realtime_factor(),
            )
            return

        # walk down from full detail and stop at the first count that fits in
        # the frame budget. fitting a line through two probes and solving was
        # tidier, but a codebook is worth about a millisecond and the probes
        # disagree by nearly as much - across two runs of the same machine the
        # same fit gave 1.09 and 0.69 ms per codebook. there are only five
        # values to choose between, so measuring them is both simpler and right.
        budget = FRAME_SECONDS / self.target_rtf
        count = self.max_codebooks
        cost = 0.0
        for count in range(self.max_codebooks, self.min_codebooks - 1, -1):
            cost, probe_state, pcm = self._probe(count, probe_state, plan)
            quality_pcm.append(pcm)
            if cost <= budget:
                break

        self.set_codebooks(count)
        self._seed_cost(cost)
        self._require_audio_quality(quality_pcm)
        self._last_tune = time.monotonic()
        log.info(
            "calibrated: %d/%d codebooks, %.1f ms/frame -> %.2fx real time "
            "(target %.2fx)",
            self.codebooks,
            self.max_codebooks,
            cost * 1000,
            self.realtime_factor(),
            self.target_rtf,
        )

    def _probe(self, codebooks: int, state, plan) -> tuple[float, object, bytes]:
        # seconds per frame at a given codebook count, measured on throwaway
        # audio nobody hears. A global warm call is made before the walk.
        self.set_codebooks(codebooks)
        started = time.monotonic()
        pcm, state = self.generate(state, plan)
        return (time.monotonic() - started) / PROBE_FRAMES, state, pcm

    def _require_audio_quality(self, chunks: list[bytes]):
        report = require_startup_pcm_quality(
            b"".join(chunks), sample_rate=SAMPLE_RATE, channels=CHANNELS
        )
        log.info(
            "startup audio passed: %.1fdBFS RMS, %.3f ZCR, %.3f flatness, "
            "%.1f%% high-band",
            report.rms_dbfs,
            report.zero_crossing_rate,
            report.spectral_flatness,
            report.high_band_fraction * 100.0,
        )

    def note_render(self, frames: int, seconds: float):
        # feed the tuner. see COST_WINDOW for why this is a median and not an
        # average: we do not want the quality flapping every time something
        # else on the machine has a moment.
        if frames <= 0 or seconds <= 0.0:
            return
        with self._cost_lock:
            self._costs.append(seconds / frames)
        self._retune(time.monotonic())

    def note_gap(self):
        # a listener's reservoir actually ran dry. that is the one measurement
        # that is not a proxy for anything, so it skips the dwell and spends a
        # codebook immediately.
        if self.pinned_codebooks or self._depth_config is None:
            return
        if self.set_codebooks(self.codebooks - 1):
            self._clear_costs()
            self._last_tune = time.monotonic()
            log.info(
                "a listener heard a gap; down to %d/%d codebooks",
                self.codebooks,
                self.max_codebooks,
            )

    def note_pressure(self):
        # Act while the client still has audio instead of waiting for a gap.
        # The dwell prevents frequent low-water reports from cascading.
        if self.pinned_codebooks or self._depth_config is None:
            return
        now = time.monotonic()
        if now - self._last_tune < TUNE_DWELL_SECONDS:
            return
        with self._cost_lock:
            sample_count = len(self._costs)
        # A stalled socket, normal first-buffer sawtooth, or throttled tab can
        # also report low water; lowering model quality cannot fix those. Do
        # not spend fidelity while measured rendering still clears its target.
        # Audible gaps remain the unconditional signal in note_gap().
        factor = self.realtime_factor()
        if sample_count < MIN_TUNE_SAMPLES or factor >= self.target_rtf:
            return
        if self.set_codebooks(self.codebooks - 1):
            self._clear_costs()
            self._last_tune = now
            log.info(
                "a listener reservoir is low; down to %d/%d codebooks",
                self.codebooks,
                self.max_codebooks,
            )

    def _retune(self, now: float):
        if self.pinned_codebooks or self._depth_config is None:
            return
        if now - self._last_tune < TUNE_DWELL_SECONDS:
            return
        with self._cost_lock:
            sample_count = len(self._costs)
        if sample_count < MIN_TUNE_SAMPLES:
            return

        factor = self.realtime_factor()
        if factor < self.target_rtf and self.codebooks > self.min_codebooks:
            changed = self.set_codebooks(self.codebooks - 1)
        elif factor > self.target_rtf * 1.3 and self.codebooks < self.max_codebooks:
            # only give detail back when there is real headroom, so the two
            # rules cannot chase each other across the same measurement
            changed = self.set_codebooks(self.codebooks + 1)
        else:
            return

        if changed:
            # Samples measured at the old loop depth cannot judge the new one.
            self._clear_costs()
            self._last_tune = now
            log.info(
                "retuned to %d/%d codebooks (%.2fx real time)",
                self.codebooks,
                self.max_codebooks,
                factor,
            )

    # --- generation ---

    def generate(self, state, plan, seed: int | None = None):
        # render one chunk, returning interleaved int16 pcm and the next state
        #
        # `plan` is a sequence of (style, cache key, guide note, frames)
        # segments. Style ramps and melodic note boundaries split it only where
        # conditioning actually changes.
        self._raise_if_stopping()
        if not self._fast:
            return self._generate_stock(state, plan)

        mx = self._mx
        sampler = self._sampler
        if state is None:
            state = sampler.get_initial_state(
                1, self._input_spec, constants={}, training=False
            )
            if seed is not None:
                streaming_state = state[0]
                _rng, previous, temporal, step = streaming_state[2]
                rng = mx.stack([mx.random.key(int(seed) & 0xFFFFFFFF)])
                seeded_decoder = (rng, previous, temporal, step)
                seeded_streaming = (
                    streaming_state[0],
                    streaming_state[1],
                    seeded_decoder,
                    streaming_state[3],
                )
                state = (seeded_streaming, *state[1:])

        outputs = []
        # the graph mlx is still working on. handing it to async_eval and only
        # blocking on it one step later lets this thread build the next frame
        # while the gpu renders this one - worth about 20% of wall clock, and
        # bit for bit the same audio as blocking on every frame.
        pending = None

        for style, key, note, frames in plan:
            self._raise_if_stopping()
            block, constants = self._conditioning(style, key, note)
            for _ in range(frames):
                if self._stop_requested.is_set():
                    # Do not leave an already-submitted GPU operation running
                    # behind teardown. Drain it, then abandon the partial PCM.
                    if pending is not None:
                        mx.eval(pending)
                    self._raise_if_stopping()
                step, state, _ = sampler.step_with_emits(
                    x=block, state=state, constants=constants, training=False
                )
                # the streaming state goes in too: left lazy it would pile up a
                # graph across the whole chunk rather than settling each frame
                mx.async_eval(
                    step.values,
                    state,
                )
                if pending is not None:
                    mx.eval(pending)
                pending = step.values
                outputs.append(step)

        if pending is not None:
            mx.eval(pending)

        # the sampler's last layer already emits int16, interleaved as
        # [frames * 1920, 2] - so this is a copy out of mlx and nothing else.
        # the old path round-tripped it through float32 and back for nothing.
        samples = np.asarray(self._sl.Sequence.concatenate_sequences(outputs).values[0])
        return np.ascontiguousarray(samples, dtype=np.int16).tobytes(), state

    def _generate_stock(self, state, plan):
        # the library's own call, one segment at a time. only used when the
        # fast path could not find what it needs.
        chunks = []
        for style, _key, note, frames in plan:
            self._raise_if_stopping()
            style_tokens = list(self._system._style_model.tokenize(style))
            style_tokens[self.style_token_levels :] = [-1] * (
                len(style_tokens) - self.style_token_levels
            )
            conditioning = {self._style_key: style_tokens}
            notes = piano_roll(note)
            if notes is not None:
                conditioning[self._notes_key] = notes
            waveform, state = self._system.generate(
                conditioning=conditioning,
                frames=frames,
                state=state,
            )
            samples = np.clip(waveform.samples, -1.0, 1.0)
            chunks.append((samples * 32767.0).astype(np.int16))

        pcm = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
        return np.ascontiguousarray(pcm).tobytes(), state

    def close(self):
        """Release model and cache state on the MLX-owning worker thread."""
        self._warm = False
        self._style_key = None
        self._notes_key = None
        self._embeddings.clear()
        self._blocks.clear()
        self._sampler = None
        self._input_spec = None
        self._depth_config = None
        self._system = None
        self._fast = False
        self._fast_sampling = False
        self._clear_costs()

        mx = getattr(self, "_mx", None)
        if mx is not None:
            try:
                mx.clear_cache()
            except Exception:  # noqa: BLE001 - best-effort native cache release
                log.exception("failed to clear MLX cache during shutdown")
