"""CT-FM whole-body adapter (project-lighter/whole_body_segmentation).

CT-FM is a SegResNet foundation model fine-tuned on the TotalSegmentator label
set (Apache-2.0 weights), runnable on CPU but slow; it is primarily a Tier-2
(GPU) entry. Because it shares TotalSegmentator's label numbering, kidney is the
union of the right- and left-kidney class ids.

Inference runs in CT-FM's own environment via ``$SEGBENCH_CTFM_PY`` (the path to
a Python interpreter with ``lighter_zoo`` + ``monai`` installed); this package
never imports torch. The adapter writes a small helper script, runs it, and reads
back a label map. Its output must pass the runner's per-case spot-check (kidney
Dice well above zero) before it is published, which guards against an
orientation or label-id mismatch silently scoring the wrong voxels.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from segbench.models.base import ModelAdapter

# CT-FM uses the TotalSegmentator v1 label numbering (kidney = 2/3, confirmed
# empirically on KiTS). Same map as the MONAI wholeBody bundle.
CTFM_LABELS: dict[str, list[int]] = {
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
import sys
import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
import torch
from lighter_zoo import SegResNet
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose, LoadImage, Orientation, Spacing, ScaleIntensityRange, EnsureType,
)

image_path, out_path = sys.argv[1], sys.argv[2]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SegResNet.from_pretrained("project-lighter/whole_body_segmentation").to(device)
model.eval()

# Preprocess to the model's working resolution (1.5 mm). Resampling is essential:
# at native resolution the 118-class logit volume is tens of GB and OOMs the GPU.
pre = Compose([
    LoadImage(ensure_channel_first=True, image_only=True),
    EnsureType(),
    Orientation(axcodes="SPL"),
    Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
    ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
])
# Stitch the sliding-window output on CPU so the full logit volume never has to
# fit in GPU memory; only the patches run on the GPU.
inferer = SlidingWindowInferer(
    roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.25,
    device=torch.device("cpu"),
)

img = pre(image_path)
spl_affine = np.asarray(img.affine)  # affine of the reoriented, resampled image
with torch.no_grad():
    logits = inferer(img.unsqueeze(0).to(device), model)
labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # (D,H,W) at 1.5mm/SPL

# Resample the label map from the SPL grid back onto the ORIGINAL image grid
# (handles orientation + any spacing difference in one nearest-neighbour step),
# so it aligns voxel-for-voxel with the ground truth.
orig = nib.load(image_path)
spl_img = nib.Nifti1Image(labels, spl_affine)
out = nibproc.resample_from_to(spl_img, (orig.shape, orig.affine), order=0)
nib.save(nib.Nifti1Image(
    np.asarray(out.dataobj).astype(np.uint8), orig.affine, orig.header), out_path)
'''


class CTFM(ModelAdapter):
    name = "CT-FM"
    license = "Apache-2.0"
    supports = frozenset(CTFM_LABELS)
    contaminated_on = frozenset({"totalsegmentator"})  # trained on TS labels

    def __init__(self, timeout: int = 7200):
        self.python = os.environ.get("SEGBENCH_CTFM_PY", "python")
        self.timeout = timeout

    def organ_masks(
        self, image: Path, out_dir: Path, organs: list[str]
    ) -> dict[str, np.ndarray]:
        out_dir.mkdir(parents=True, exist_ok=True)
        label_path = out_dir / "ctfm_label.nii.gz"
        if not (label_path.exists() and label_path.stat().st_size > 0):
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False
            ) as fh:
                fh.write(_HELPER)
                helper = fh.name
            subprocess.run(
                [self.python, helper, str(image), str(label_path)],
                check=True, timeout=self.timeout,
            )

        lbl = np.asarray(nib.load(str(label_path)).dataobj)
        masks: dict[str, np.ndarray] = {}
        for organ in organs:
            ids = CTFM_LABELS.get(organ)
            if ids:
                masks[organ] = np.isin(lbl, ids)
        return masks
