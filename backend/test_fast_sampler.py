"""Equivalence checks for the specialized MRT2 top-k sampler."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class FastSamplerTests(unittest.TestCase):
    def test_specialized_sampler_matches_ranges_and_cfg_shape(self):
        try:
            import magenta_rt  # noqa: F401 - installs sequence_layers hook
            import mlx.core as mx
            import sequence_layers.mlx as sl
            from magenta_rt.mlx import depthformer
            from fast_sampler import install

            rng = np.random.default_rng(7)
            values = rng.normal(size=(1, 1, 12_294)).astype(np.float32)
            logits = sl.Sequence.from_values(mx.array(values))
            key = mx.stack([mx.random.key(123)])
            temperature = mx.array([0.0])
            top_k = mx.array([50], dtype=mx.int32)
            original = depthformer._sample_categorical_with_temperature

            expected = []
            for low in (6, 1030, 5126, 11_270):
                result = original(
                    logits, temperature, top_k, None, key, None, 0, (low, low + 1024)
                )
                mx.eval(result.values)
                expected.append(int(np.asarray(result.values)[0, 0]))

            self.assertTrue(install(depthformer, mx, 50))
            optimized = depthformer._sample_categorical_with_temperature
            actual = []
            for low in (6, 1030, 5126, 11_270):
                result = optimized(
                    logits, temperature, top_k, None, key, None, 0, (low, low + 1024)
                )
                mx.eval(result.values)
                actual.append(int(np.asarray(result.values)[0, 0]))

            self.assertEqual(actual, expected)
            self.assertTrue(
                all(
                    low <= value < low + 1024
                    for low, value in zip((6, 1030, 5126, 11_270), actual)
                )
            )

            for seed in range(20):
                stochastic_key = mx.stack([mx.random.key(seed)])
                for low in (6, 5126, 11_270):
                    for candidate_top_k, top_p in (
                        (top_k, None),
                        (None, mx.array([0.9])),
                    ):
                        result = optimized(
                            logits,
                            mx.array([1.1]),
                            candidate_top_k,
                            top_p,
                            stochastic_key,
                            None,
                            0,
                            (low, low + 1024),
                        )
                        mx.eval(result.values)
                        value = int(np.asarray(result.values)[0, 0])
                        self.assertTrue(low <= value < low + 1024)

            cfg_logits = sl.Sequence.from_values(mx.array(np.repeat(values, 2, axis=0)))
            cfg_keys = mx.stack([mx.random.key(1), mx.random.key(2)])
            cfg_result = optimized(
                cfg_logits,
                mx.array([1.1, 1.1]),
                mx.array([50, 50], dtype=mx.int32),
                None,
                cfg_keys,
                mx.array([0.0, 1.6]),
                1,
                (2054, 3078),
            )
            mx.eval(cfg_result.values)
            cfg_values = np.asarray(cfg_result.values)[:, 0]
            self.assertEqual(cfg_values.shape, (2,))
            self.assertEqual(cfg_values[0], cfg_values[1])
            self.assertTrue(np.all((cfg_values >= 2054) & (cfg_values < 3078)))
        except RuntimeError as exc:
            # MLX needs a functioning Metal compiler/device. Keep ordinary test
            # discovery useful on CI and while another process owns Metal.
            if "metal" in str(exc).lower() or "xpc_error" in str(exc).lower():
                self.skipTest(f"MLX Metal runtime unavailable: {exc}")
            raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
