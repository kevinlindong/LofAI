"""Tests for the native MusicCoCa RVQ tokenizer and interpreter release."""

import functools
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import style_tokens  # noqa: E402


def _reference_rvq(codebooks: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Plain float64 residual nearest-neighbour quantization."""
    residual = vector.astype(np.float64)
    codes = []
    for level in range(codebooks.shape[0]):
        distances = np.sum((codebooks[level].astype(np.float64) - residual) ** 2, axis=-1)
        code = int(np.argmin(distances))
        codes.append(code)
        residual = residual - codebooks[level, code]
    return np.array(codes, dtype=np.int32)


def _synthetic_codebooks(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = 0.5 ** np.arange(style_tokens.RVQ_LEVELS)
    return (
        rng.normal(size=(style_tokens.RVQ_LEVELS, style_tokens.CODEBOOK_SIZE, style_tokens.EMBEDDING_DIM))
        * scales[:, None, None]
    ).astype(np.float32)


class NativeTokenizerTests(unittest.TestCase):
    def test_matches_a_plain_residual_quantizer(self):
        codebooks = _synthetic_codebooks()
        tokenizer = style_tokens.NativeStyleTokenizer(codebooks)
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(6, style_tokens.EMBEDDING_DIM)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        tokens = tokenizer.tokenize(vectors)
        self.assertEqual(tokens.shape, (6, 12))
        self.assertEqual(tokens.dtype, np.int32)
        for vector, row in zip(vectors, tokens):
            np.testing.assert_array_equal(row, _reference_rvq(codebooks, vector))

    def test_single_vector_and_batch_shapes(self):
        tokenizer = style_tokens.NativeStyleTokenizer(_synthetic_codebooks())
        vector = np.ones(style_tokens.EMBEDDING_DIM, dtype=np.float32)
        self.assertEqual(tokenizer.tokenize(vector).shape, (12,))
        self.assertEqual(tokenizer.tokenize(np.stack([vector, vector])).shape, (2, 12))
        self.assertEqual(tokenizer.tokenize(vector.reshape(1, 1, -1)).shape, (1, 1, 12))
        with self.assertRaises(ValueError):
            tokenizer.tokenize(np.ones(10, dtype=np.float32))

    def test_rejects_malformed_codebooks(self):
        with self.assertRaises(ValueError):
            style_tokens.NativeStyleTokenizer(np.zeros((12, 16, 768), dtype=np.float32))
        broken = _synthetic_codebooks()
        broken[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            style_tokens.NativeStyleTokenizer(broken)

    def test_cache_round_trip_preserves_tokens(self):
        codebooks = _synthetic_codebooks()
        tokenizer = style_tokens.NativeStyleTokenizer(codebooks)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rvq.npz"
            style_tokens._save_cache(path, tokenizer)
            loaded = style_tokens._load_cache(path)
        self.assertIsNotNone(loaded)
        vector = np.random.default_rng(2).normal(size=style_tokens.EMBEDDING_DIM).astype(np.float32)
        np.testing.assert_array_equal(loaded.tokenize(vector), tokenizer.tokenize(vector))


class _FakeStyleModel:
    """Mimics MusicCoCa's lazily built, cached_property-backed interpreters."""

    built: list[str] = []

    def __init__(self, resource_dir=None):
        self._resource_dir = resource_dir

    @functools.cached_property
    def _text_encoder(self):
        self.built.append("text")
        return object()

    @functools.cached_property
    def _mapper(self):
        return object()

    @functools.cached_property
    def _quantizer(self):
        return object()

    @functools.cached_property
    def _audio_preprocessor(self):
        return object()

    @functools.cached_property
    def _music_encoder(self):
        return object()


class ReleaseTests(unittest.TestCase):
    def test_release_drops_only_built_interpreters_and_allows_rebuild(self):
        model = _FakeStyleModel()
        model._text_encoder  # noqa: B018 - build two of the five
        model._quantizer  # noqa: B018

        released = style_tokens.release_interpreters(model, include_quantizer=False)
        self.assertEqual(released, ["_text_encoder"])
        self.assertIn("_quantizer", vars(model))

        released = style_tokens.release_interpreters(model, include_quantizer=True)
        self.assertEqual(released, ["_quantizer"])
        self.assertNotIn("_text_encoder", vars(model))

        # A later request transparently rebuilds the interpreter it needs.
        before = len(_FakeStyleModel.built)
        model._text_encoder  # noqa: B018
        self.assertEqual(len(_FakeStyleModel.built), before + 1)

    def test_load_or_extract_without_a_quantizer_file_keeps_tflite(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _FakeStyleModel(resource_dir=tmp)
            self.assertIsNone(style_tokens.load_or_extract(model, Path(tmp), None))
        self.assertIsNone(style_tokens.load_or_extract(_FakeStyleModel(), None, None))

    def test_cached_tokenizer_is_trusted_without_the_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_dir = Path(tmp) / "musiccoca"
            resource_dir.mkdir()
            quantizer = resource_dir / style_tokens.QUANTIZER_FILE
            quantizer.write_bytes(b"not a real model")
            model = _FakeStyleModel(resource_dir=resource_dir)
            cache_dir = Path(tmp) / "cache"
            cache_path = style_tokens._cache_path(cache_dir, quantizer)
            style_tokens._save_cache(cache_path, style_tokens.NativeStyleTokenizer(_synthetic_codebooks()))

            tokenizer = style_tokens.load_or_extract(model, cache_dir, None)
            self.assertIsNotNone(tokenizer)
            # No interpreter was ever built to serve a cache hit.
            self.assertNotIn("_quantizer", vars(model))

            # Touching the model file invalidates the cache key.
            quantizer.write_bytes(b"different bytes")
            self.assertNotEqual(style_tokens._cache_path(cache_dir, quantizer), cache_path)


class EngineIntegrationTests(unittest.TestCase):
    def test_engine_prefers_the_native_tokenizer_for_conditioning(self):
        from engine import MRTEngine

        class Tokenizer:
            def __init__(self):
                self.calls = 0

            def tokenize(self, _style):
                self.calls += 1
                return np.arange(12, dtype=np.int32)[::-1]

        class StyleModel:
            def tokenize(self, _style):
                raise AssertionError("TFLite tokenizer must not be used")

        class System:
            _style_model = StyleModel()

            def _build_conditioning(self, conditioning, *args):
                return dict(conditioning), {}

        engine = MRTEngine()
        engine._style_key, engine._notes_key, engine._drums_key = "style", "notes", "drums"
        engine._system = System()
        engine._tokenizer = Tokenizer()
        block, _ = engine._conditioning(np.zeros(768, dtype=np.float32), None)
        self.assertEqual(block["style"], list(range(11, -1, -1)))
        self.assertEqual(engine._tokenizer.calls, 1)
        # Fixed stations are tokenized once and cached under their key.
        engine._conditioning(np.zeros(768, dtype=np.float32), "station")
        engine._conditioning(np.zeros(768, dtype=np.float32), "station", drum=0)
        self.assertEqual(engine._tokenizer.calls, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
