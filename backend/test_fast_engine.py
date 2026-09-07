"""Guard and equivalence checks for the specialized MRT2 streaming step."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class FastEngineGuardTests(unittest.TestCase):
    def test_refuses_unpinned_library_versions(self):
        import fast_engine

        with patch.object(
            fast_engine.importlib.metadata, "version", return_value="2.0.4"
        ):
            self.assertIsNone(fast_engine.install(object(), None, None))

    def test_refuses_unfamiliar_sampler_shapes(self):
        import fast_engine

        class BareSystem:
            _sampler = object()  # no .layers at all

        with patch.object(
            fast_engine.importlib.metadata, "version", return_value="2.0.3"
        ):
            self.assertIsNone(fast_engine.install(BareSystem(), None, None))

    def test_tree_has_arrays_only_reports_real_arrays(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX unavailable")
        from fast_engine import _tree_has_arrays

        self.assertFalse(_tree_has_arrays(((), {}, [(), ()]), mx))
        self.assertFalse(_tree_has_arrays((0, "x", None), mx))
        self.assertTrue(_tree_has_arrays(((), (mx.zeros(1),)), mx))
        self.assertTrue(_tree_has_arrays({"a": [((mx.zeros(1),),)]}, mx))


class TruncationDecodeFixTests(unittest.TestCase):
    def test_skipped_codebooks_no_longer_contribute_code_zero_centroids(self):
        """The RVQ decode must sum only active codebooks.

        Upstream pads skipped codebooks with code 0 and codes_to_embeddings
        sums that codebook's row 0 - a full-magnitude centroid - into every
        frame. After the fix the decode of a 10-active frame equals the
        decode of the same tokens truncated to 10 columns.
        """
        try:
            import magenta_rt  # noqa: F401 - installs the vendored sl hook
            import mlx.core as mx
            import sequence_layers.mlx as sl
            from magenta_rt.mlx.spectrostream.modeling import (
                ResidualVectorQuantizer,
            )
            import fast_engine
        except ImportError as exc:
            self.skipTest(f"magenta-rt runtime unavailable: {exc}")
        except RuntimeError as exc:
            if "metal" in str(exc).lower():
                self.skipTest(f"MLX Metal runtime unavailable: {exc}")
            raise

        quantizer = ResidualVectorQuantizer(
            ResidualVectorQuantizer.Config(
                num_quantizers=12,
                num_embeddings=16,
                embedding_dim=8,
                use_unique_codes=False,
                beta=0.0,
                dynamic_masking=False,
                target_num_quantizers=(12,),
                full_quantizer_dropout_rate=0.0,
            )
        )
        rng = np.random.default_rng(3)
        quantizer.embedding = mx.array(
            rng.normal(size=(12, 16, 8)).astype(np.float32)
        )

        class Config:
            num_active_codebooks = 10
            num_codebooks = 12

        class Lambda:
            _fn = quantizer.codes_to_embeddings

        class Spectro:
            pass

        class Depthformer:
            pass

        class Sampler:
            layers = [None, None, Lambda(), None, None]
            spectrostream = Spectro()

        sampler = Sampler()
        sampler.spectrostream.quantizer = quantizer

        self.assertTrue(
            fast_engine._install_truncation_fix(sampler, Config, sl)
        )

        codes = rng.integers(0, 16, size=(1, 3, 10)).astype(np.int32)
        padded = np.concatenate(
            [codes, np.zeros((1, 3, 2), dtype=np.int32)], axis=-1
        )
        fixed = sampler.layers[2]._fn(sl.Sequence.from_values(mx.array(padded)))
        expected = quantizer.codes_to_embeddings(
            sl.Sequence.from_values(mx.array(codes))
        )
        contaminated = quantizer.codes_to_embeddings(
            sl.Sequence.from_values(mx.array(padded))
        )
        mx.eval(fixed.values, expected.values, contaminated.values)

        np.testing.assert_array_equal(
            np.asarray(fixed.values), np.asarray(expected.values)
        )
        # The stock behavior really was different (code-0 centroids added).
        self.assertFalse(
            np.array_equal(
                np.asarray(contaminated.values), np.asarray(expected.values)
            )
        )

        # At full depth the fix is inert: identical to the original function.
        Config.num_active_codebooks = 12
        full = sampler.layers[2]._fn(sl.Sequence.from_values(mx.array(padded)))
        stock = quantizer.codes_to_embeddings(
            sl.Sequence.from_values(mx.array(padded))
        )
        mx.eval(full.values, stock.values)
        np.testing.assert_array_equal(
            np.asarray(full.values), np.asarray(stock.values)
        )

    def test_truncation_fix_requires_the_expected_wiring(self):
        import fast_engine

        class Lambda:
            _fn = None

        class Sampler:
            layers = [None, None, Lambda(), None, None]

        class Config:
            num_active_codebooks = None
            num_codebooks = 12

        sampler = Sampler()
        sampler.spectrostream = type("S", (), {"quantizer": object()})()
        self.assertFalse(
            fast_engine._install_truncation_fix(sampler, Config, None)
        )


class SlicedProjectorTests(unittest.TestCase):
    def test_sliced_projection_matches_full_rows_within_one_ulp(self):
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import fast_engine
        except ImportError as exc:
            self.skipTest(f"MLX unavailable: {exc}")

        try:
            vocab, features, codebooks, size, reserved = 3078, 64, 3, 1024, 6

            class Config:
                num_reserved_tokens = reserved
                codebook_size = size
                num_codebooks = codebooks

            linear = nn.Linear(features, vocab, bias=True)
            linear.update(
                {
                    "weight": mx.random.normal((vocab, features), key=mx.random.key(1)),
                    "bias": mx.random.normal((vocab,), key=mx.random.key(2)),
                }
            )
            quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)

            class Inner:
                _linear = quantized
                compute_dtype = mx.bfloat16
                _param_dtype = mx.float32

            class Deferred:
                inner = Inner()

            # Native nanobind functions have no inspectable signature on
            # MLX 0.32.2. This must not silently disable the entire fast step.
            with patch.object(
                fast_engine.inspect, "signature", side_effect=ValueError("native callable")
            ):
                projector = fast_engine._build_projector(Deferred(), Config, mx)
            self.assertIsNotNone(projector)

            probe = mx.random.normal((1, 1, features), key=mx.random.key(3)).astype(
                mx.bfloat16
            )
            full = quantized(probe)
            for index in range(codebooks):
                low = reserved + index * size
                sliced = projector(probe, index)
                reference = np.asarray(
                    full[..., low : low + size].astype(mx.float32)
                )
                candidate = np.asarray(sliced.astype(mx.float32))
                tolerance = np.maximum(np.abs(reference), 1.0) * 2.0**-8
                self.assertTrue(
                    np.all(np.abs(reference - candidate) <= tolerance),
                    f"codebook {index} deviates beyond one bf16 ulp",
                )
        except RuntimeError as exc:
            if "metal" in str(exc).lower() or "xpc_error" in str(exc).lower():
                self.skipTest(f"MLX Metal runtime unavailable: {exc}")
            raise

    def test_unfamiliar_projection_layers_are_refused(self):
        try:
            import mlx.core as mx
            import fast_engine
        except ImportError as exc:
            self.skipTest(f"MLX unavailable: {exc}")

        class Config:
            num_reserved_tokens = 6
            codebook_size = 1024
            num_codebooks = 12

        class NoInner:
            inner = None

        self.assertIsNone(fast_engine._build_projector(NoInner(), Config, mx))

        class OddLinear:
            _linear = object()  # neither Linear nor QuantizedLinear

        class Inner:
            inner = OddLinear()

        # attribute layout exists but the linear type is unknown
        OddLinear.compute_dtype = None
        OddLinear._param_dtype = None
        self.assertIsNone(fast_engine._build_projector(Inner(), Config, mx))


class SynthesisWindowCacheTests(unittest.TestCase):
    def test_window_is_computed_once_and_preserves_streaming_audio(self):
        try:
            import magenta_rt  # noqa: F401
            import mlx.core as mx
            import sequence_layers.mlx as sl
            from sequence_layers.mlx import signal
            from fast_engine import _cache_synthesis_windows
        except ImportError as exc:
            self.skipTest(f"MLX runtime unavailable: {exc}")
        from functools import wraps
        from types import SimpleNamespace

        window_fn = signal.inverse_stft_window_fn(4)
        calls = []

        @wraps(window_fn)
        def counted_window(*args, **kwargs):
            calls.append(1)
            return window_fn(*args, **kwargs)

        layer = sl.InverseSTFT(
            frame_length=16, frame_step=4, fft_length=16,
            window_fn=counted_window,
        )
        rng = np.random.default_rng(8)
        values = rng.normal(size=(1, 6, 9, 2)) + 1j * rng.normal(size=(1, 6, 9, 2))
        inputs = sl.Sequence.from_values(mx.array(values.astype(np.complex64)))

        def render():
            state = layer.get_initial_state(1, inputs.channel_spec)
            outputs = []
            for index in range(inputs.shape[1]):
                output, state = layer.step(inputs[:, index:index + 1], state)
                outputs.append(output)
            return np.asarray(sl.Sequence.concatenate_sequences(outputs).values)

        expected = render()
        self.assertEqual(len(calls), 6)
        spectrostream = SimpleNamespace(named_modules=lambda: [("decoder.istft", layer)])
        self.assertTrue(_cache_synthesis_windows(spectrostream, sl))
        self.assertEqual(len(calls), 7)  # Prewarmed during installation.
        actual = render()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(calls), 7)
        self.assertTrue(_cache_synthesis_windows(spectrostream, sl))
        self.assertEqual(len(calls), 7)

    def test_arbitrary_window_functions_are_not_cached(self):
        try:
            import magenta_rt  # noqa: F401
            import sequence_layers.mlx as sl
            from fast_engine import _cache_synthesis_windows
        except ImportError as exc:
            self.skipTest(f"MLX runtime unavailable: {exc}")
        from types import SimpleNamespace

        window = lambda size: np.ones(size, dtype=np.float32)
        layer = sl.InverseSTFT(
            frame_length=16, frame_step=4, fft_length=16, window_fn=window
        )
        spectrostream = SimpleNamespace(named_modules=lambda: [("decoder.istft", layer)])
        self.assertFalse(_cache_synthesis_windows(spectrostream, sl))
        self.assertIs(layer._window_fn, window)


if __name__ == "__main__":
    unittest.main(verbosity=2)
