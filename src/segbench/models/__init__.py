"""Model adapters and their registry.

Adapters are constructed from a config entry: ``{type: totalsegmentator, ...}``.
Inference itself runs in each model's own isolated environment (see the module
docstrings); this package only orchestrates and reads the resulting NIfTI files.
"""

from __future__ import annotations

from segbench.models.base import ModelAdapter
from segbench.models.ctfm import CTFM
from segbench.models.monai_bundle import MONAIWholeBody
from segbench.models.moose import MOOSE
from segbench.models.totalsegmentator import TotalSegmentator

MODELS: dict[str, type[ModelAdapter]] = {
    "totalsegmentator": TotalSegmentator,
    "ctfm": CTFM,
    "monai": MONAIWholeBody,
    "moose": MOOSE,
}


def build_model(spec: dict[str, object]) -> ModelAdapter:
    """Build an adapter from a config dict with a ``type`` key."""
    spec = dict(spec)
    kind = str(spec.pop("type"))
    if kind not in MODELS:
        raise KeyError(f"unknown model type {kind!r}; have {sorted(MODELS)}")
    return MODELS[kind](**spec)  # type: ignore[arg-type]


__all__ = [
    "CTFM",
    "MODELS",
    "MOOSE",
    "MONAIWholeBody",
    "ModelAdapter",
    "TotalSegmentator",
    "build_model",
]
