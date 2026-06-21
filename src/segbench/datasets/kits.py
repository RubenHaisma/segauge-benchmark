"""KiTS23 adapter — the seed dataset.

KiTS23 (Kidney Tumor Segmentation Challenge 2023) is downloadable **per case**,
which makes it the practical choice for a reproducible seed run: you fetch only
the handful of cases you score, not a multi-gigabyte archive. TotalSegmentator
and the other whole-body models were not trained on KiTS, so kidney/tumor
scores on it are a *fair* comparison for them.

Imaging volumes live on the Hugging Face mirror used by the official `kits23`
downloader; segmentations live in the kits23 GitHub repository. Neither is
re-hosted here — only the derived scores are published.

License: CC BY-NC-SA 4.0 (non-commercial). Publishing derived metric tables is
fine; the leaderboard stays non-commercial and cites the dataset.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

from segbench.datasets.base import CaseRecord, Dataset
from segbench.schema import KITS23_KIDNEY_SCHEMA, KITS23_SCHEMA

IMAGING_URL = (
    "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/"
    "resolve/main/images/case_{n:05d}.nii.gz"
)
SEG_URL = (
    "https://raw.githubusercontent.com/neheller/kits23/main/"
    "dataset/case_{n:05d}/segmentation.nii.gz"
)

CITATION = (
    "Heller et al., The KiTS23 Challenge (2023). "
    "github.com/neheller/kits23"
)


def _download(url: str, dst: Path, timeout: int = 600) -> None:
    """Atomic, idempotent download: skip if present, write via a temp file."""
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".partial.{dst.name}"
    req = urllib.request.Request(url, headers={"User-Agent": "segauge-benchmark"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dst)


class KiTS23(Dataset):
    name = "kits23"
    license = "CC-BY-NC-SA-4.0"
    citation = CITATION
    url = "https://github.com/neheller/kits23"

    def __init__(
        self,
        root: Path,
        case_numbers: list[int] | None = None,
        whole_kidney: bool = True,
    ) -> None:
        super().__init__(root)
        # Default seed set: the first few training cases (public ground truth).
        self.case_numbers = case_numbers if case_numbers is not None else list(range(5))
        # Whole-kidney view (the fair comparison for general organ models): score
        # against the union of kidney/tumor/cyst, not parenchyma alone.
        self.whole_kidney = whole_kidney
        self.schema = KITS23_KIDNEY_SCHEMA if whole_kidney else KITS23_SCHEMA

    def _selected(self, n: int | None) -> list[int]:
        nums = self.case_numbers
        return nums if n is None else nums[:n]

    def ensure_available(self, n: int | None = None) -> None:
        for num in self._selected(n):
            case_dir = self.root / f"case_{num:05d}"
            _download(IMAGING_URL.format(n=num), case_dir / "imaging.nii.gz")
            _download(SEG_URL.format(n=num), case_dir / "segmentation.nii.gz")

    def _whole_kidney_gt(self, seg_path: Path) -> Path:
        """Derive a binary whole-kidney label map (union of 1/2/3) -> id 1."""
        out = seg_path.parent / "segmentation_whole_kidney.nii.gz"
        if out.exists() and out.stat().st_size > 0:
            return out
        img = nib.load(str(seg_path))
        whole = (np.asarray(img.dataobj) > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(whole, img.affine, img.header), str(out))
        return out

    def cases(self, n: int | None = None) -> list[CaseRecord]:
        records: list[CaseRecord] = []
        for num in self._selected(n):
            case_id = f"case_{num:05d}"
            case_dir = self.root / case_id
            seg = case_dir / "segmentation.nii.gz"
            gt = self._whole_kidney_gt(seg) if self.whole_kidney else seg
            records.append(
                CaseRecord(
                    case_id=case_id,
                    image=case_dir / "imaging.nii.gz",
                    gt=gt,
                    metadata={"modality": "CT", "dataset": "kits23"},
                )
            )
        return records
