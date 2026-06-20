"""MONAI Bundle adapter (wholeBody_ct_segmentation).

Runs MONAI's own whole-body CT bundle (the de-facto framework's reference model,
Apache-2.0) in its environment via ``$SEGBENCH_MONAI_PY``. The bundle's
postprocessing inverts orientation + spacing, so its label map is written back on
the input grid (no resampling needed here). We use the 3 mm low-resolution model
by default (~6 GB VRAM) so it fits commodity GPUs; the 1.5 mm high-res model
needs ~29 GB. Label numbering follows TotalSegmentator v1 (kidney = ids 2 and 3).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from segbench.models.base import ModelAdapter

# canonical organ -> MONAI wholeBody label id(s)
MONAI_LABELS: dict[str, list[int]] = {
    "kidney": [2, 3],
    "right_kidney": [2],
    "left_kidney": [3],
    "spleen": [1],
    "liver": [5],
    "stomach": [6],
    "pancreas": [10],
    "gallbladder": [4],
    "aorta": [7],
    "esophagus": [42],
    "duodenum": [56],
    "inferior_vena_cava": [8],
    "right_adrenal_gland": [11],
    "left_adrenal_gland": [12],
    "bladder": [104],
}

_HELPER = '''\
import os, sys
import torch
from monai.bundle import download, run

bundle_dir, image_path, out_dir, highres = sys.argv[1:5]
root = os.path.join(bundle_dir, "wholeBody_ct_segmentation")
if not os.path.isdir(root):
    download(name="wholeBody_ct_segmentation", bundle_dir=bundle_dir)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
run(
    config_file=os.path.join(root, "configs/inference.json"),
    bundle_root=root,
    datalist=[image_path],
    output_dir=out_dir,
    highres=(highres == "true"),
    device=device,
)
'''


class MONAIWholeBody(ModelAdapter):
    name = "MONAI-wholeBody"
    license = "Apache-2.0"
    supports = frozenset(MONAI_LABELS)
    contaminated_on = frozenset({"totalsegmentator"})  # bundle trained on TS labels

    def __init__(self, highres: bool = False, timeout: int = 7200):
        self.python = os.environ.get("SEGBENCH_MONAI_PY", "python")
        self.bundle_dir = os.environ.get("SEGBENCH_MONAI_BUNDLE_DIR", "/weights/monai")
        self.highres = highres
        self.timeout = timeout

    def organ_masks(
        self, image: Path, out_dir: Path, organs: list[str]
    ) -> dict[str, np.ndarray]:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image.name.replace(".nii.gz", "").replace(".nii", "")
        label_path = out_dir / stem / f"{stem}_trans.nii.gz"
        if not (label_path.exists() and label_path.stat().st_size > 0):
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                fh.write(_HELPER)
                helper = fh.name
            subprocess.run(
                [
                    self.python, helper, self.bundle_dir, str(image), str(out_dir),
                    "true" if self.highres else "false",
                ],
                check=True, timeout=self.timeout,
            )

        lbl = np.asarray(nib.load(str(label_path)).dataobj)
        masks: dict[str, np.ndarray] = {}
        for organ in organs:
            ids = MONAI_LABELS.get(organ)
            if ids:
                masks[organ] = np.isin(lbl, ids)
        return masks
