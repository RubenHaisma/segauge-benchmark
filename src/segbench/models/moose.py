"""MOOSE adapter (moosez, clin_ct_organs).

MOOSE comes from the nuclear-medicine / PET-CT world (ENHANCE-PET) - a genuinely
independent lineage from the TotalSegmentator family, which makes it a useful
contrast on the leaderboard. It runs via ``$SEGBENCH_MOOSE_PY`` and writes a
single multilabel NIfTI resampled back onto the input grid (no resampling needed
here). The ``clin_ct_organs`` model uses kidney_left = 6, kidney_right = 7.
Weights are CC-BY-4.0 (scores publishable).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from segbench.models.base import ModelAdapter

# canonical organ -> moosez clin_ct_organs label id(s)
MOOSE_LABELS: dict[str, list[int]] = {
    "kidney": [6, 7],
    "left_kidney": [6],
    "right_kidney": [7],
    "spleen": [15],
    "liver": [8],
    "stomach": [16],
    "pancreas": [14],
    "gallbladder": [5],
    "bladder": [3],
    "left_adrenal_gland": [1],
    "right_adrenal_gland": [2],
}

_HELPER = '''\
import os, sys, glob, shutil, tempfile


def main():
    import torch
    from moosez import moose

    src, out_dir, final = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()
    # MOOSE expects a CT-prefixed NIfTI; copy the input under that convention.
    ct = os.path.join(tmp, "CT_case.nii.gz")
    shutil.copy(src, ct)
    acc = "cuda" if torch.cuda.is_available() else "cpu"
    moose(ct, "clin_ct_organs", out_dir, acc)
    hits = glob.glob(
        os.path.join(out_dir, "**", "*organs_segmentation*.nii.gz"), recursive=True
    )
    if not hits:
        hits = glob.glob(
            os.path.join(out_dir, "**", "*segmentation*.nii.gz"), recursive=True
        )
    if not hits:
        raise SystemExit("MOOSE produced no segmentation output")
    shutil.copy(sorted(hits)[-1], final)


if __name__ == "__main__":
    main()
'''


class MOOSE(ModelAdapter):
    name = "MOOSE"
    license = "CC-BY-4.0"
    supports = frozenset(MOOSE_LABELS)
    contaminated_on = frozenset()  # Dataset123_Organs, not KiTS/AMOS

    def __init__(self, timeout: int = 7200):
        self.python = os.environ.get("SEGBENCH_MOOSE_PY", "python")
        self.timeout = timeout

    def organ_masks(
        self, image: Path, out_dir: Path, organs: list[str]
    ) -> dict[str, np.ndarray]:
        out_dir.mkdir(parents=True, exist_ok=True)
        label_path = out_dir / "moose_label.nii.gz"
        if not (label_path.exists() and label_path.stat().st_size > 0):
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                fh.write(_HELPER)
                helper = fh.name
            subprocess.run(
                [self.python, helper, str(image), str(out_dir / "moose_out"),
                 str(label_path)],
                check=True, timeout=self.timeout,
            )

        lbl = np.asarray(nib.load(str(label_path)).dataobj)
        masks: dict[str, np.ndarray] = {}
        for organ in organs:
            ids = MOOSE_LABELS.get(organ)
            if ids:
                masks[organ] = np.isin(lbl, ids)
        return masks
