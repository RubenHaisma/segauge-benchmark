"""TotalSegmentator adapter.

Runs the `TotalSegmentator` CLI (the de-facto reference for CT organ
segmentation, Apache-2.0) in its own environment and reads the per-structure
NIfTI files it writes. Reading the *named* output files avoids coupling to
TotalSegmentator's internal label numbering. On CPU we pass ``--roi_subset`` so
only the structures we actually score are computed, and ``--fast`` for the 3 mm
model, which keeps a kidney run to a couple of minutes per case.

The executable is taken from ``$SEGBENCH_TS_BIN`` (or ``TotalSegmentator`` on
PATH), so the heavy environment lives outside this package.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from segbench.models.base import ModelAdapter

# canonical organ -> TotalSegmentator `total`-task structure file name(s)
TS_STRUCTURES: dict[str, list[str]] = {
    "kidney": ["kidney_left", "kidney_right"],
    "left_kidney": ["kidney_left"],
    "right_kidney": ["kidney_right"],
    "spleen": ["spleen"],
    "liver": ["liver"],
    "stomach": ["stomach"],
    "pancreas": ["pancreas"],
    "gallbladder": ["gallbladder"],
    "aorta": ["aorta"],
    "esophagus": ["esophagus"],
    "duodenum": ["duodenum"],
    "inferior_vena_cava": ["inferior_vena_cava"],
    "right_adrenal_gland": ["adrenal_gland_right"],
    "left_adrenal_gland": ["adrenal_gland_left"],
    "bladder": ["urinary_bladder"],
}


class TotalSegmentator(ModelAdapter):
    name = "TotalSegmentator"
    license = "Apache-2.0"
    supports = frozenset(TS_STRUCTURES)
    contaminated_on = frozenset({"totalsegmentator"})  # its own training set

    def __init__(
        self,
        fast: bool = True,
        device: str = "cpu",
        timeout: int = 3600,
        name: str | None = None,
    ):
        self.binary = os.environ.get("SEGBENCH_TS_BIN", "TotalSegmentator")
        self.fast = fast
        self.device = device
        self.timeout = timeout
        # Distinct display name per config so the full and 3 mm fast models can
        # appear as separate leaderboard entries.
        self.name = name or ("TotalSegmentator-fast" if fast else "TotalSegmentator")

    def version(self) -> str:
        try:
            out = subprocess.run(
                [self.binary, "--version"], capture_output=True, text=True, timeout=60
            )
            return (out.stdout or out.stderr).strip() or "unknown"
        except Exception:
            return "unknown"

    def organ_masks(
        self, image: Path, out_dir: Path, organs: list[str]
    ) -> dict[str, np.ndarray]:
        seg_dir = out_dir / "ts_raw"
        seg_dir.mkdir(parents=True, exist_ok=True)

        needed: list[str] = []
        for organ in organs:
            needed.extend(TS_STRUCTURES.get(organ, []))
        needed = sorted(set(needed))

        cmd = [
            self.binary, "-i", str(image), "-o", str(seg_dir),
            "--device", self.device, "--quiet",
        ]
        if self.fast:
            cmd.append("--fast")
        if needed:
            cmd += ["--roi_subset", *needed]
        subprocess.run(cmd, check=True, timeout=self.timeout)

        masks: dict[str, np.ndarray] = {}
        for organ in organs:
            parts = []
            for struct in TS_STRUCTURES.get(organ, []):
                f = seg_dir / f"{struct}.nii.gz"
                if f.exists():
                    parts.append(np.asarray(nib.load(str(f)).dataobj) > 0)
            if parts:
                merged = parts[0]
                for p in parts[1:]:
                    merged = merged | p
                masks[organ] = merged
        return masks
