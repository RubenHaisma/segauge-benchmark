"""AMOS22 adapter — the Tier-2 multi-organ target.

AMOS22 is the fair, CC-BY-4.0, multi-organ (15 abdominal structures) benchmark
for the full leaderboard run. TotalSegmentator was not trained on AMOS, so it is
a fair external test. AMOS ships as a single ~24 GB archive, so unlike KiTS it is
not practical to fetch per case; this adapter therefore expects an already
downloaded AMOS directory in the standard layout::

    {root}/imagesTr/amos_XXXX.nii.gz
    {root}/labelsTr/amos_XXXX.nii.gz

and scores the public-label training split (no submission server needed). It is
used for the Tier-2 GPU run, not the CPU seed.
"""

from __future__ import annotations

from pathlib import Path

from segbench.datasets.base import CaseRecord, Dataset
from segbench.schema import AMOS_CT_SCHEMA

CITATION = (
    "Ji et al., AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for "
    "Versatile Medical Image Segmentation, NeurIPS 2022 (Datasets & Benchmarks)."
)


def _modality_for(case_num: int) -> str:
    # AMOS convention: CT cases are numbered below 500, MRI at/above 500.
    return "CT" if case_num < 500 else "MR"


class AMOS22(Dataset):
    name = "amos_ct"
    schema = AMOS_CT_SCHEMA
    license = "CC-BY-4.0"
    citation = CITATION
    url = "https://amos22.grand-challenge.org/"

    def __init__(self, root: Path, modality: str = "CT") -> None:
        super().__init__(root)
        self.modality = modality

    def _data_dir(self) -> Path:
        """Locate the dir containing imagesTr/ (handles the amos22/ nesting)."""
        if (self.root / "imagesTr").is_dir():
            return self.root
        for p in sorted(self.root.glob("*/imagesTr")):
            return p.parent
        return self.root

    def ensure_available(self, n: int | None = None) -> None:
        images = self._data_dir() / "imagesTr"
        if not images.is_dir():
            raise FileNotFoundError(
                f"AMOS not found at {self.root}. AMOS is a single large archive; "
                "download it from Zenodo (record 7155725) and unpack it so that "
                f"{images} exists. This adapter does not auto-download AMOS."
            )

    def cases(self, n: int | None = None) -> list[CaseRecord]:
        data = self._data_dir()
        images = data / "imagesTr"
        labels = data / "labelsTr"
        records: list[CaseRecord] = []
        for img in sorted(images.glob("amos_*.nii.gz")):
            case_id = img.name.replace(".nii.gz", "")
            case_num = int(case_id.split("_")[1])
            if _modality_for(case_num) != self.modality:
                continue
            gt = labels / img.name
            if not gt.exists():
                continue
            records.append(
                CaseRecord(
                    case_id=case_id,
                    image=img,
                    gt=gt,
                    metadata={"modality": _modality_for(case_num), "dataset": "amos"},
                )
            )
            if n is not None and len(records) >= n:
                break
        return records
