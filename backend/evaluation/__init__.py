"""Reproducible render and listening evaluation tools for lofAI.

The package deliberately stays separate from the live server.  Importing it
does not load Magenta RealTime, MLX, or a model checkpoint, which keeps matrix
planning, signal analysis, reports, and their tests inexpensive.
"""

from .audio_metrics import METRICS_DISCLAIMER, AudioMetrics, analyze_audio
from .spec import EvaluationSpec, RenderCase, load_evaluation_spec

__all__ = [
    "AudioMetrics",
    "EvaluationSpec",
    "METRICS_DISCLAIMER",
    "RenderCase",
    "analyze_audio",
    "load_evaluation_spec",
]
