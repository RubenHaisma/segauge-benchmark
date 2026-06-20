"""Shared synthetic fixtures: small 3D label volumes with anisotropic spacing."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def save_label(path: Path, arr: np.ndarray, spacing=(1.0, 1.0, 3.0)) -> Path:
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), affine), str(path))
    return path


def cube(shape=(40, 40, 40), lo=10, hi=30, label=1) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[lo:hi, lo:hi, lo:hi] = label
    return arr


@pytest.fixture
def kidney_case(tmp_path: Path):
    """An image + a ground-truth kidney cube (label 1), anisotropic spacing."""
    gt = cube(label=1)
    img = np.zeros((40, 40, 40), dtype=np.int16)
    image = save_label(tmp_path / "imaging.nii.gz", img)
    gtruth = save_label(tmp_path / "segmentation.nii.gz", gt)
    return image, gtruth
