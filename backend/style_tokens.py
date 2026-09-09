"""Native replica of MusicCoCa's residual vector quantizer.

MusicCoCa turns a 768-dimensional style embedding into 12 tokens with a
TFLite graph (``pretrained_vector_quantizer.tflite``). Reading that graph
shows a plain residual nearest-neighbour quantizer: at each of 12 levels it
computes ``|r|^2 - 2 r.c_i + |c_i|^2`` against a 1,024-entry codebook via a
fully connected layer, takes the argmin, gathers the chosen code, and
subtracts it from the residual. Nothing else: no projection, no
normalisation.

Extracting the codebooks once and running the same arithmetic in NumPy
reproduced the TFLite tokens exactly on every vector tried locally (the
station prompts, 48 style-ramp blends, 500 random unit vectors, and 400
near-tie perturbations). The extraction re-verifies itself against the
interpreter before the result is trusted or cached, and any doubt falls back
to the TFLite path.

Why bother: the TFLite interpreter and its XNNPACK delegate otherwise stay
resident for the life of the process on the model thread, and style ramps
invoke it for every blended embedding. The NumPy version is about as fast,
needs no interpreter, and lets the engine release every MusicCoCa TFLite
model after startup. On an 8 GB machine that is real headroom.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RVQ_LEVELS = 12
CODEBOOK_SIZE = 1024
EMBEDDING_DIM = 768
QUANTIZER_FILE = "pretrained_vector_quantizer.tflite"
CACHE_FORMAT = b"lofai-musiccoca-rvq-v1"


class NativeStyleTokenizer:
    """12-level residual VQ over 768-d embeddings, matching MusicCoCa's TFLite."""

    def __init__(self, codebooks: np.ndarray, norms: np.ndarray | None = None):
        codebooks = np.ascontiguousarray(codebooks, dtype=np.float32)
        if codebooks.shape != (RVQ_LEVELS, CODEBOOK_SIZE, EMBEDDING_DIM):
            raise ValueError(f"unexpected codebook shape {codebooks.shape}")
        if not np.isfinite(codebooks).all():
            raise ValueError("codebooks contain non-finite values")
        self.codebooks = codebooks
        # The TFLite fully connected layer carries -2c as its kernel and |c|^2
        # as its bias. Keep the bias values it shipped with when available so
        # the distance arithmetic rounds identically; -2c is exact in float.
        if norms is None:
            norms = np.sum(codebooks.astype(np.float64) ** 2, axis=-1).astype(np.float32)
        norms = np.ascontiguousarray(norms, dtype=np.float32)
        if norms.shape != (RVQ_LEVELS, CODEBOOK_SIZE):
            raise ValueError(f"unexpected norm shape {norms.shape}")
        self.norms = norms

    def tokenize(self, embedding) -> np.ndarray:
        """Return int32 tokens: shape [12] for one vector, [..., 12] for a batch."""
        vectors = np.asarray(embedding, dtype=np.float32)
        if vectors.shape[-1] != EMBEDDING_DIM:
            raise ValueError(f"expected {EMBEDDING_DIM}-d embeddings, got {vectors.shape}")
        flat = vectors.reshape(-1, EMBEDDING_DIM)
        tokens = np.empty((flat.shape[0], RVQ_LEVELS), dtype=np.int32)
        codebooks, norms = self.codebooks, self.norms
        for row, vector in enumerate(flat):
            residual = vector.copy()
            for level in range(RVQ_LEVELS):
                # Same three terms as the TFLite graph, added in its order:
                # the fully connected layer yields r.(-2c) + |c|^2, then |r|^2
                # is added. Scaling by -2 is exact in floating point, so this
                # equals a matmul against the stored -2c kernel term for term;
                # the extraction step verifies the tokens agree regardless.
                distances = (
                    np.float32(-2.0) * (residual @ codebooks[level].T) + norms[level]
                ) + np.float32(np.dot(residual, residual))
                code = int(np.argmin(distances))
                tokens[row, level] = code
                residual = residual - codebooks[level, code]
        return tokens.reshape(vectors.shape[:-1] + (RVQ_LEVELS,))


def _quantizer_path(style_model) -> Path | None:
    resource_dir = getattr(style_model, "_resource_dir", None)
    if resource_dir is None:
        return None
    path = Path(resource_dir) / QUANTIZER_FILE
    return path if path.is_file() else None


def _cache_path(cache_dir: Path | None, quantizer: Path) -> Path | None:
    if cache_dir is None:
        return None
    try:
        stat = quantizer.stat()
    except OSError:
        return None
    digest = hashlib.sha256()
    digest.update(CACHE_FORMAT + b"\0")
    digest.update(f"{quantizer.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return cache_dir / f"style-rvq-{digest.hexdigest()[:24]}.npz"


def extract_from_tflite(quantizer_path: Path) -> NativeStyleTokenizer:
    """Read the codebooks out of the RVQ TFLite graph.

    Walks the graph in execution order: every level is a FULLY_CONNECTED
    (kernel = -2c, bias = |c|^2) followed later by a GATHER over that level's
    codebook constant. The final level has no GATHER (no further residual),
    so its codebook is recovered from the kernel as ``-kernel / 2``, which is
    exact in floating point.
    """
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(quantizer_path))
    interpreter.allocate_tensors()
    ops = interpreter._get_ops_details()  # noqa: SLF001 - no public graph API

    kernels: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    gathered: list[np.ndarray] = []
    for op in ops:
        name = op["op_name"]
        inputs = [int(index) for index in op["inputs"]]
        if name == "FULLY_CONNECTED":
            kernels.append(np.array(interpreter.get_tensor(inputs[1]), dtype=np.float32))
            biases.append(np.array(interpreter.get_tensor(inputs[2]), dtype=np.float32))
        elif name == "GATHER":
            gathered.append(np.array(interpreter.get_tensor(inputs[0]), dtype=np.float32))

    if len(kernels) != RVQ_LEVELS or len(gathered) != RVQ_LEVELS - 1:
        raise ValueError(
            f"unfamiliar RVQ graph: {len(kernels)} projections, {len(gathered)} gathers"
        )
    codebooks = []
    for level in range(RVQ_LEVELS):
        derived = -0.5 * kernels[level]
        if level < RVQ_LEVELS - 1 and not np.array_equal(derived, gathered[level]):
            raise ValueError(f"RVQ level {level}: kernel and codebook disagree")
        codebooks.append(derived)
    return NativeStyleTokenizer(np.stack(codebooks), np.stack(biases))


def _verify(tokenizer: NativeStyleTokenizer, style_model, vectors: np.ndarray) -> bool:
    if vectors.size == 0:
        return True
    expected = np.asarray(style_model.tokenize(vectors.astype(np.float32)), dtype=np.int32)
    actual = tokenizer.tokenize(vectors)
    if actual.shape != expected.shape:
        return False
    return bool(np.array_equal(actual, expected))


def _verification_vectors(known: list[np.ndarray] | None) -> np.ndarray:
    rng = np.random.default_rng(20260909)
    random = rng.normal(size=(64, EMBEDDING_DIM)).astype(np.float32)
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    parts = [random]
    if known:
        stations = np.stack([np.asarray(vector, dtype=np.float32) for vector in known])
        parts.append(stations)
        # Ramps blend station pairs; the exact points of the standard ramp.
        blends = []
        for a in range(len(stations)):
            for b in range(len(stations)):
                if a != b:
                    for t in (0.04296875, 0.31640625, 0.68359375, 0.95703125):
                        blends.append((1.0 - t) * stations[a] + t * stations[b])
        if blends:
            parts.append(np.stack(blends).astype(np.float32))
    return np.concatenate(parts)


def _load_cache(path: Path) -> NativeStyleTokenizer | None:
    try:
        with np.load(path, allow_pickle=False) as cached:
            codebooks = cached["codebooks"]
            norms = cached["norms"]
        return NativeStyleTokenizer(codebooks, norms)
    except (OSError, ValueError, KeyError) as exc:
        log.warning("ignoring invalid style tokenizer cache %s: %s", path, exc)
        return None


def _save_cache(path: Path, tokenizer: NativeStyleTokenizer):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=path.parent, delete=False
        ) as temp:
            temp_path = Path(temp.name)
            np.savez(temp, codebooks=tokenizer.codebooks, norms=tokenizer.norms)
        os.replace(temp_path, path)
    except OSError as exc:
        log.warning("could not cache the style tokenizer: %s", exc)


def load_or_extract(
    style_model,
    cache_dir: Path | None,
    known_embeddings: list[np.ndarray] | None = None,
) -> NativeStyleTokenizer | None:
    """Return a verified native tokenizer, or None to keep using TFLite.

    A cached copy is trusted without loading the interpreter; that is the
    whole point on a warm start. A fresh extraction is checked token-for-token
    against the TFLite quantizer on random vectors, the station embeddings,
    and their ramp blends before it is used or written to disk.
    """
    quantizer = _quantizer_path(style_model)
    if quantizer is None:
        return None
    cache_path = _cache_path(cache_dir, quantizer)
    if cache_path is not None and cache_path.is_file():
        tokenizer = _load_cache(cache_path)
        if tokenizer is not None:
            log.info("loaded the native style tokenizer from cache")
            return tokenizer
    try:
        tokenizer = extract_from_tflite(quantizer)
    except Exception as exc:  # noqa: BLE001 - keep the TFLite path on any doubt
        log.warning("native style tokenizer unavailable; keeping TFLite: %s", exc)
        return None
    vectors = _verification_vectors(known_embeddings)
    try:
        verified = _verify(tokenizer, style_model, vectors)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not verify the native style tokenizer: %s", exc)
        return None
    if not verified:
        log.warning(
            "native style tokenizer disagreed with TFLite on %d test vectors; keeping TFLite",
            len(vectors),
        )
        return None
    log.info(
        "native style tokenizer verified against TFLite on %d vectors", len(vectors)
    )
    if cache_path is not None:
        _save_cache(cache_path, tokenizer)
    return tokenizer


# MusicCoCa builds each TFLite interpreter lazily through functools.cached_property
# and stores it in the instance dict under these names. Removing the entries
# frees the interpreters; a later call simply rebuilds the one it needs.
_LAZY_INTERPRETERS = (
    "_text_encoder",
    "_mapper",
    "_audio_preprocessor",
    "_music_encoder",
)
_LAZY_QUANTIZER = "_quantizer"


def release_interpreters(style_model, *, include_quantizer: bool) -> list[str]:
    """Drop MusicCoCa's resident TFLite interpreters; return what was released."""
    store = getattr(style_model, "__dict__", None)
    if store is None:
        return []
    names = list(_LAZY_INTERPRETERS) + ([_LAZY_QUANTIZER] if include_quantizer else [])
    released = [name for name in names if store.pop(name, None) is not None]
    if released:
        import gc

        gc.collect()
    return released
