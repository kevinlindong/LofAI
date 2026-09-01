"""Target-range sampling for Magenta RT 2's eager MLX decoder.

MRT2 emits one shared 12,294-token logit vector for each of 12 residual
codebooks, but a depth step may select from only one contiguous 1,024-token
range. magenta-rt 2.0.3 masks the other 11 ranges, then sorts and generates
Gumbel noise for the entire vector. This specialization slices the valid range
first. It implements the same categorical distribution with a different (and
documented) deterministic RNG trajectory because fewer random values are drawn.

The dependency is pinned and the signature checked because this deliberately
specializes a private hot path.
"""

import importlib.metadata
import inspect


EXPECTED_PARAMETERS = (
    "logits",
    "temperature",
    "top_k",
    "top_p",
    "rng_key",
    "classifier_free_guidance_scale",
    "classifier_free_guidance_arity",
    "valid_range",
)


def install(depthformer_module, mx, configured_top_k: int) -> bool:
    original = depthformer_module._sample_categorical_with_temperature
    if getattr(original, "_lofai_fast_sampler", False):
        return True

    try:
        version = importlib.metadata.version("magenta-rt")
        parameters = tuple(inspect.signature(original).parameters)
    except (importlib.metadata.PackageNotFoundError, TypeError, ValueError):
        return False
    if (
        version != "2.0.3"
        or parameters != EXPECTED_PARAMETERS
        or configured_top_k < 1
    ):
        return False

    def sample(
        logits,
        temperature,
        top_k,
        top_p,
        rng_key,
        classifier_free_guidance_scale,
        classifier_free_guidance_arity,
        valid_range,
    ):
        if valid_range is None:
            return original(
                logits,
                temperature,
                top_k,
                top_p,
                rng_key,
                classifier_free_guidance_scale,
                classifier_free_guidance_arity,
                valid_range,
            )

        low, high = valid_range
        if not (0 <= low < high <= logits.shape[-1]):
            return original(
                logits,
                temperature,
                top_k,
                top_p,
                rng_key,
                classifier_free_guidance_scale,
                classifier_free_guidance_arity,
                valid_range,
            )

        # Everything outside this contiguous codebook span would be replaced
        # with -inf by the generic function. Remove it before both sorting and
        # Gumbel generation, then restore the shared-vocabulary offset.
        sliced_logits = logits.apply_values(lambda values: values[..., low:high])
        sampled = original(
            sliced_logits,
            temperature,
            top_k,
            top_p,
            rng_key,
            classifier_free_guidance_scale,
            classifier_free_guidance_arity,
            None,
        )
        return sampled.apply_values(
            lambda values: values + mx.array(low, dtype=values.dtype)
        )

    sample._lofai_fast_sampler = True
    sample._lofai_original = original
    depthformer_module._sample_categorical_with_temperature = sample
    return True
