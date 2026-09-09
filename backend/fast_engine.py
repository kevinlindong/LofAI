"""Hot-loop specialization of Magenta RT 2's eager MLX streaming step.

magenta-rt 2.0.3 rebuilds several per-frame constants inside
``StreamingEncoderDecoderSampler.step`` and projects every depth step onto
the full 12,294-token vocabulary even though a depth step may only sample
from its own codebook's contiguous 1,024-token span. This module installs an
instance-level replacement for the live no-CFG, batch-of-one configuration
lofAI actually runs (CFG is encoded as conditioning tokens, so the batch is
never expanded):

- ``to_logits`` slicing: each depth step multiplies against only its
  codebook's 1,024 weight rows instead of all 12,294, reading 12x less
  weight data on the projection that runs 10-12 times per frame. Metal
  picks a different kernel tiling for the smaller matrix, so logits can
  differ from the stock path in the last bf16 mantissa bit; install-time
  verification requires agreement within one ulp and refuses otherwise.
  Like the fast sampler's documented RNG-trajectory change, this trades
  bit-identical takes for the same distribution rendered faster.
- conditioning-encoder cache: the encoder is stateless in this
  configuration, so its output per conditioning block is a constant that is
  computed once and reused for every frame of the block (exactly equal).
- constant hoisting: the depth transformer's initial state, the
  skipped-codebook dummy tokens, and the CFG/delay bookkeeping that never
  varies on the live path are built once instead of per frame (exactly
  equal).
- synthesis-window cache: the codec's inverse-STFT window is constant, but
  upstream rebuilds it on the GPU and copies it to NumPy every frame. Reuse
  the original window so that CPU readback no longer synchronizes each
  streaming step in the middle of the asynchronous frame pipeline (exactly
  equal).

It also repairs quality truncation. Upstream pads skipped codebooks with
each codebook's code 0, and the SpectroStream RVQ then *sums those
full-magnitude centroid vectors* into every decoded frame (measured norms of
code 0 match the codebook mean, e.g. |q10[0]| = 6.24 vs mean 6.23). Proper
RVQ truncation adds nothing for skipped levels, so the decode path here
slices the token frame to the active count before ``codes_to_embeddings``.
At 12 active codebooks the fix is structurally inert.

Like ``fast_sampler``, this deliberately specializes a pinned private hot
path: the dependency version and every structural assumption are checked at
install, unfamiliar callers fall back to the stock step per call, and
``MRT_FAST_ENGINE=0`` disables the whole module.
"""

import importlib.metadata
import inspect
import logging
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

log = logging.getLogger(__name__)

_CFG_SCALE_PREFIX = "classifier_free_guidance_scale"

# Signatures this specialization replicates. If upstream moves, refuse.
EXPECTED_STEP_PARAMETERS = (
    "self",
    "x",
    "state",
    "forced_tokens",
    "training",
    "constants",
)
EXPECTED_VOCAB = 12_294
EXPECTED_CODEBOOK_SIZE = 1_024
EXPECTED_RESERVED = 6
EXPECTED_CODEBOOKS = 12

_ENCODER_CACHE_SIZE = 32


@dataclass
class FastEngineStatus:
    """What actually got installed, for /health and logs."""

    step_specialized: bool = False
    sliced_logits: bool = False
    cached_encoder: bool = False
    truncation_decode_fix: bool = False
    cached_synthesis_window: bool = False

    def summary(self) -> str:
        parts = []
        if self.step_specialized:
            parts.append("specialized step")
        if self.sliced_logits:
            parts.append("sliced logits")
        if self.cached_encoder:
            parts.append("cached conditioning encoder")
        if self.truncation_decode_fix:
            parts.append("clean RVQ truncation")
        if self.cached_synthesis_window:
            parts.append("cached synthesis window")
        return ", ".join(parts) if parts else "nothing installed"


def _tree_has_arrays(tree, mx) -> bool:
    if isinstance(tree, mx.array):
        return True
    if isinstance(tree, (tuple, list)):
        return any(_tree_has_arrays(item, mx) for item in tree)
    if isinstance(tree, dict):
        return any(_tree_has_arrays(item, mx) for item in tree.values())
    return False


def _build_projector(to_logits, decoder_config, mx):
    """Return a per-codebook sliced projection exactly equal to `to_logits`.

    The final depth layer is a DenseDeferred wrapping a Dense backed by
    ``nn.Linear`` (or ``nn.QuantizedLinear`` after ``nn.quantize``). Weight,
    scale, and bias rows are laid out along the output axis, so slicing rows
    [low, high) yields the same values for exactly those logits. Returns
    ``None`` when the structure is unfamiliar or the equality check fails.
    """
    import mlx.nn as nn

    inner = getattr(to_logits, "inner", None)
    linear = getattr(inner, "_linear", None)
    if inner is None or linear is None:
        return None

    compute_dtype = inner.compute_dtype or inner._param_dtype
    reserved = decoder_config.num_reserved_tokens
    size = decoder_config.codebook_size
    vocab = decoder_config.num_codebooks * size + reserved

    slices = []
    if isinstance(linear, nn.QuantizedLinear):
        weight, scales, biases = linear.weight, linear.scales, linear.biases
        if weight.shape[0] != vocab:
            return None
        group_size, bits = linear.group_size, linear.bits
        # MLX exposes quantized_matmul through nanobind; inspect.signature
        # raises on the pinned 0.32.2 build. Mirror QuantizedLinear's own
        # mode attribute instead: versions with this attribute pass it to
        # the native call, while older affine-only versions omit the keyword.
        mode = getattr(linear, "mode", None)
        matmul_kwargs = {"mode": mode} if mode is not None else {}
        add_bias = "bias" in linear
        bias = linear["bias"] if add_bias else None
        for index in range(decoder_config.num_codebooks):
            low = reserved + index * size
            high = low + size
            entry = (
                weight[low:high],
                scales[low:high],
                biases[low:high],
                bias[low:high] if add_bias else None,
            )
            mx.eval(*(part for part in entry if part is not None))
            slices.append(entry)

        def project(values, index):
            w, s, b, bias_slice = slices[index]
            y = mx.quantized_matmul(
                values.astype(compute_dtype),
                w,
                scales=s,
                biases=b,
                transpose=True,
                group_size=group_size,
                bits=bits,
                **matmul_kwargs,
            )
            if bias_slice is not None:
                y = y + bias_slice
            return y

    elif isinstance(linear, nn.Linear):
        weight = linear.weight
        if weight.shape[0] != vocab:
            return None
        add_bias = "bias" in linear
        bias = linear["bias"] if add_bias else None
        for index in range(decoder_config.num_codebooks):
            low = reserved + index * size
            high = low + size
            entry = (
                weight[low:high],
                bias[low:high] if add_bias else None,
            )
            mx.eval(*(part for part in entry if part is not None))
            slices.append(entry)

        def project(values, index):
            w, bias_slice = slices[index]
            x = values.astype(compute_dtype)
            if bias_slice is not None:
                # nn.Linear uses addmm when a bias is present; mirror it so
                # sliced results round identically.
                return mx.addmm(bias_slice, x, w.T)
            return x @ w.T

    else:
        return None

    # Sanity proof on random inputs: the sliced projection must produce the
    # same values as the corresponding rows of the full projection, within
    # one bf16 ulp. Metal selects different kernel tilings for a 12,294-row
    # and a 1,024-row weight matrix, so per-element accumulation order - and
    # therefore the last mantissa bit - occasionally differs. That bounded
    # rounding difference is the documented cost of reading 12x less weight
    # data; anything larger means the slicing is wrong and we refuse it.
    in_features = (
        linear.scales.shape[1] * linear.group_size
        if isinstance(linear, nn.QuantizedLinear)
        else linear.weight.shape[1]
    )
    for seed in (0, 1, 2):
        probe = mx.random.normal((1, 1, in_features), key=mx.random.key(seed)).astype(
            compute_dtype
        )
        full = linear(probe.astype(compute_dtype))
        for index in (0, decoder_config.num_codebooks - 1):
            low = reserved + index * size
            sliced = project(probe, index)
            reference = full[..., low : low + size].astype(mx.float32)
            candidate = sliced.astype(mx.float32)
            # One bf16 ulp is 2^-8 of the value's magnitude.
            tolerance = mx.maximum(mx.abs(reference), 1.0) * (2.0**-8)
            if not mx.all(mx.abs(reference - candidate) <= tolerance).item():
                log.info(
                    "sliced to_logits deviates beyond one bf16 ulp from the "
                    "stock projection on this MLX build; keeping the full "
                    "projection"
                )
                return None
    return project


def _stateless_encoder(layer0, system, mx, sl) -> bool:
    """True when the conditioning encoder carries no step state."""
    try:
        spec = sl.ChannelSpec(shape=(system._num_channels,), dtype=mx.int32)
        state = layer0.encoder.body.get_initial_state(
            1, spec, training=False, constants={}
        )
    except Exception:  # noqa: BLE001 - structure probe only
        return False
    return not _tree_has_arrays(state, mx)


def install(system, mx, sl) -> FastEngineStatus | None:
    """Specialize the live sampler in place. Returns status, or None."""
    try:
        version = importlib.metadata.version("magenta-rt")
    except importlib.metadata.PackageNotFoundError:
        return None
    if version != "2.0.3":
        log.info("fast engine targets magenta-rt 2.0.3, found %s; skipping", version)
        return None

    from magenta_rt.mlx import depthformer as df

    sampler = system._sampler
    layers = getattr(sampler, "layers", None)
    if not layers or len(layers) != 5:
        return None
    layer0 = layers[0]
    if not isinstance(layer0, df.StreamingEncoderDecoderSampler):
        return None
    decoder = layer0.decoder
    config = decoder.config

    step_parameters = tuple(
        inspect.signature(df.StreamingEncoderDecoderSampler.step.__wrapped__
                          if hasattr(df.StreamingEncoderDecoderSampler.step, "__wrapped__")
                          else df.StreamingEncoderDecoderSampler.step).parameters
    )
    decoder_parameters = tuple(
        inspect.signature(df.MultivariateDecoder.step_with_emits).parameters
    )
    structural = (
        step_parameters == EXPECTED_STEP_PARAMETERS
        and decoder_parameters == EXPECTED_STEP_PARAMETERS
        and layer0.streaming_encoder
        and layer0.encoder_lookahead == 0
        and layer0.conditioning_name is not None
        and config.num_codebooks == EXPECTED_CODEBOOKS
        and config.codebook_size == EXPECTED_CODEBOOK_SIZE
        and config.num_reserved_tokens == EXPECTED_RESERVED
        and config.soft_cap_logits is not None
        and len(getattr(decoder.depth_body, "layers", ())) == 4
    )
    if not structural:
        log.info("fast engine found an unfamiliar sampler structure; skipping")
        return None

    status = FastEngineStatus()
    status.truncation_decode_fix = _install_truncation_fix(sampler, config, sl)
    status.cached_synthesis_window = _cache_synthesis_windows(
        sampler.spectrostream, sl
    )

    prefix_layers = list(decoder.depth_body.layers[:-1])
    projector = _build_projector(decoder.depth_body.layers[-1], config, mx)
    status.sliced_logits = projector is not None
    cache_encoder = _stateless_encoder(layer0, system, mx, sl)
    status.cached_encoder = cache_encoder

    original_step = layer0.step
    soft_cap = float(config.soft_cap_logits)
    conditioning_name = layer0.conditioning_name
    mean_in_f32 = df._mean_in_f32

    encoder_cache: OrderedDict = OrderedDict()
    depth_state_cache: dict = {}
    dummy_cache: dict = {}

    def encode_block(x, encoder_state, constants):
        key = (id(x), id(constants))
        hit = encoder_cache.get(key)
        if hit is not None and hit[0] is x and hit[1] is constants:
            encoder_cache.move_to_end(key)
            return hit[2]
        encoded, _ = layer0.encoder.body.step(
            x, encoder_state, training=False, constants=constants
        )
        # Hold the block and constants so the ids stay live and unambiguous.
        encoder_cache[key] = (x, constants, encoded)
        if len(encoder_cache) > _ENCODER_CACHE_SIZE:
            encoder_cache.popitem(last=False)
        return encoded

    def initial_depth_state(spec):
        key = (tuple(spec.shape), spec.dtype)
        state = depth_state_cache.get(key)
        if state is None:
            states = []
            probe = spec
            for layer in prefix_layers:
                states.append(
                    layer.get_initial_state(1, probe, training=False, constants=None)
                )
                probe = layer.get_output_spec(probe, constants=None)
            state = tuple(states)
            depth_state_cache[key] = state
        return state

    def depth_prefix_step(x, state):
        new_states = []
        for layer, layer_state in zip(prefix_layers, state):
            x, layer_state, _ = layer.step_with_emits(
                x, layer_state, training=False, constants=None
            )
            new_states.append(layer_state)
        return x, tuple(new_states)

    def dummy_tokens(index, shape, dtype):
        key = (index, shape, dtype)
        value = dummy_cache.get(key)
        if value is None:
            value = mx.full(shape, EXPECTED_RESERVED + index * EXPECTED_CODEBOOK_SIZE, dtype=dtype)
            mx.eval(value)
            dummy_cache[key] = value
        return value

    def sample_step(self, encoded, state, constants):
        """Sample one frame of tokens from already-encoded conditioning.

        ``state`` is the sampler layer's own state tuple. The conditioning
        encoder has already run (``encoded``), so this half of the step is a
        pure function of arrays: that is what lets ``compiled_engine`` trace
        it with ``mx.compile`` while the eager path keeps the encoder cache.
        """
        encoder_state, _previous_output, sampler_state, delay_countdown = state
        sampler_constants = dict(constants)
        sampler_constants[conditioning_name] = encoded

        # --- MultivariateDecoder.step_with_emits, minus per-frame constants ---
        # The decoder ignores its step input and embeds the previous frame
        # held in its own state; mirror that exactly.
        rng, previous_frame, temporal_state, step_count = sampler_state
        embedded = decoder.embedder.layer(previous_frame)
        temporal_inputs = embedded.apply_values(mean_in_f32, axis=-2)
        temporal_outputs, temporal_state = decoder.temporal_body.step(
            temporal_inputs,
            temporal_state,
            training=False,
            constants=sampler_constants,
        )

        depth_state = initial_depth_state(temporal_outputs.channel_spec)
        depth_inputs = temporal_outputs
        active = config.num_active_codebooks or config.num_codebooks
        sample_fn = df._sample_categorical_with_temperature
        temperature = constants.get("temperature")
        top_k = constants.get("top_k")
        top_p = constants.get("top_p", None)
        samples = []

        for index in range(active):
            hidden, depth_state = depth_prefix_step(depth_inputs, depth_state)
            low = EXPECTED_RESERVED + index * EXPECTED_CODEBOOK_SIZE
            if projector is not None:
                logits = hidden.apply_values(lambda v, i=index: projector(v, i))
                logits = logits.apply_values(
                    lambda v: mx.tanh(v / soft_cap) * soft_cap
                )
                logits = logits.apply_values_masked(lambda v: v.astype(mx.float32))
                sample = sample_fn(
                    logits, temperature, top_k, top_p, rng, None, 0, None
                )
                sample = sample.apply_values(
                    lambda v: v + mx.array(low, dtype=v.dtype)
                )
            else:
                logits, _, _ = decoder.depth_body.layers[-1].step_with_emits(
                    hidden, (), training=False, constants=None
                )
                logits = logits.apply_values(
                    lambda v: mx.tanh(v / soft_cap) * soft_cap
                )
                logits = logits.apply_values_masked(lambda v: v.astype(mx.float32))
                sample = sample_fn(
                    logits, temperature, top_k, top_p, rng, None, 0,
                    (low, low + EXPECTED_CODEBOOK_SIZE),
                )
            # One key per batch element; batch is guarded to one.
            rng = mx.random.split(rng[0])[1][None]
            samples.append(sample.values)
            depth_inputs = decoder.embedder.layer(sample)

        first = samples[0]
        for index in range(active, config.num_codebooks):
            samples.append(dummy_tokens(index, first.shape, first.dtype))

        tokens = sl.Sequence.from_values(mx.stack(samples, axis=-1))
        sampler_state = (rng, tokens, temporal_state, step_count + 1)

        # encoder_lookahead == 0: the delay countdown is identically zero, so
        # the upstream where() plumbing reduces to pass-through.
        return tokens, (encoder_state, tokens, sampler_state, delay_countdown)

    def fast_step(self, x, state, *, forced_tokens=None, training=False, constants=None):
        constants = constants or {}
        if (
            training
            or forced_tokens is not None
            or x.shape[0] != 1
            or any(key.startswith(_CFG_SCALE_PREFIX) for key in constants)
        ):
            return original_step(
                x, state, forced_tokens=forced_tokens, training=training,
                constants=constants,
            )

        encoder_state = state[0]
        if cache_encoder:
            encoded = encode_block(x, encoder_state, constants)
        else:
            encoded, encoder_state = self.encoder.body.step(
                x, encoder_state, training=False, constants=constants
            )
            state = (encoder_state, *state[1:])
        return sample_step(self, encoded, state, constants)

    import types as types_module

    layer0.step = types_module.MethodType(fast_step, layer0)
    layer0._lofai_fast_engine = status
    layer0._lofai_original_step = original_step
    # The two halves of the step, for callers that want to run the encoder
    # once per conditioning block and trace only the sampling half.
    layer0._lofai_encode = encode_block
    layer0._lofai_sample_step = types_module.MethodType(sample_step, layer0)
    status.step_specialized = True
    log.info("fast engine installed: %s", status.summary())
    return status


def _cache_synthesis_windows(spectrostream, sl) -> bool:
    """Compute the codec's immutable inverse-STFT window once per instance.

    SequenceLayers' inverse_stft_window_fn rebuilds this constant on the GPU
    and copies it to NumPy on every frame. That copy synchronizes the GPU in
    the middle of sampler.step, defeating asynchronous frame evaluation.
    Reusing the original window removes that barrier without changing its
    arithmetic, the overlap-add state, or a single decoded sample.
    """
    installed = False
    for _, layer in spectrostream.named_modules():
        if not isinstance(layer, sl.InverseSTFT):
            continue
        window_fn = layer._window_fn
        if getattr(window_fn, "_lofai_cached_window", False):
            installed = True
            continue
        # Cache only the known pure function, never an arbitrary user window
        # whose output could change between calls.
        if (
            getattr(window_fn, "__module__", None) != "sequence_layers.mlx.signal"
            or getattr(window_fn, "__name__", None) != "inverse_stft_window_fn_inner"
        ):
            continue
        cached = lru_cache(maxsize=4)(window_fn)
        cached(layer._frame_length)
        cached._lofai_cached_window = True
        layer._window_fn = cached
        installed = True
    return installed


def _install_truncation_fix(sampler, decoder_config, sl) -> bool:
    """Slice skipped codebooks out of the decode instead of summing code 0."""
    quantizer = sampler.spectrostream.quantizer
    codes_layer = sampler.layers[2]
    original = getattr(codes_layer, "_fn", None)
    if original is None or getattr(original, "_lofai_truncation_fix", False):
        return getattr(original, "_lofai_truncation_fix", False)
    bound = getattr(original, "__func__", None)
    if bound is not getattr(type(quantizer), "codes_to_embeddings", None):
        return False

    def truncated_codes_to_embeddings(codes):
        active = decoder_config.num_active_codebooks or decoder_config.num_codebooks
        if active < codes.shape[2]:
            codes = codes.apply_values_masked(lambda v: v[:, :, :active])
        return original(codes)

    truncated_codes_to_embeddings._lofai_truncation_fix = True
    codes_layer._fn = truncated_codes_to_embeddings
    return True
