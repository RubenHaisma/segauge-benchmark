"""Model adapter contract.

Every model emits its own label numbering and its own file layout. An adapter's
job is to hide that: given an input image it returns one binary mask per
*canonical organ name* it supports, on the input grid. The base class assembles
those into a single label map in the target dataset's schema and caches it, so a
model is run at most once per case.

Two rules keep the leaderboard honest:

- A model is only scored on the organs it *supports* (``supports``). A general
  organ model that cannot segment kidney tumors is shown "n/a" for tumor, never
  a silent zero.
- A model declares which datasets it was trained on (``contaminated_on``); the
  runner marks those (model, dataset) cells as contaminated and excludes them
  from ranking.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from segbench.schema import LabelSchema, normalize_organ


def _resample_nn(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Nearest-neighbour resample a binary mask to ``shape`` (safety fallback)."""
    if mask.shape == shape:
        return mask
    factors = [s / m for s, m in zip(shape, mask.shape, strict=True)]
    out = zoom(mask.astype(np.uint8), factors, order=0)
    return out.astype(bool)


class ModelAdapter:
    """Base class for model adapters."""

    name: str
    license: str
    #: Canonical organ names this model can produce.
    supports: frozenset[str] = frozenset()
    #: Dataset names this model was trained on (scores on them are contaminated).
    contaminated_on: frozenset[str] = frozenset()

    def version(self) -> str:
        return "unknown"

    def is_fair_on(self, dataset_name: str) -> bool:
        return dataset_name not in self.contaminated_on

    def supported_in(self, schema: LabelSchema) -> list[str]:
        """Canonical organ keys this model supports that exist in ``schema``."""
        keys = {o.key for o in schema.organs}
        return [k for k in (normalize_organ(s) for s in self.supports) if k in keys]

    def organ_masks(
        self, image: Path, out_dir: Path, organs: list[str]
    ) -> dict[str, np.ndarray]:  # pragma: no cover - subclass I/O
        """Run inference; return {canonical_organ_key: bool array on image grid}."""
        raise NotImplementedError

    def predict(
        self, image: Path, gt: Path, out_dir: Path, schema: LabelSchema
    ) -> Path:
        """Produce a cached canonical-schema label map aligned to the GT grid."""
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / f"{self.name}__{schema.name}.nii.gz"
        if cache.exists() and cache.stat().st_size > 0:
            return cache

        organs = self.supported_in(schema)
        masks = self.organ_masks(image, out_dir, organs)

        gt_img = nib.load(str(gt))
        ref_shape = gt_img.shape
        out = np.zeros(ref_shape, dtype=np.uint8)
        by_key = {o.key: o for o in schema.organs}
        for key, mask in masks.items():
            organ = by_key.get(key)
            if organ is None:
                continue
            out[_resample_nn(np.asarray(mask, dtype=bool), ref_shape)] = organ.id

        nib.save(nib.Nifti1Image(out, gt_img.affine, gt_img.header), str(cache))
        return cache
