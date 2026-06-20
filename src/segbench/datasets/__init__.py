"""Dataset adapters and their registry."""

from __future__ import annotations

from pathlib import Path

from segbench.datasets.amos import AMOS22
from segbench.datasets.base import CaseRecord, Dataset
from segbench.datasets.kits import KiTS23

DATASETS: dict[str, type[Dataset]] = {
    "kits23": KiTS23,
    "amos_ct": AMOS22,
}


def get_dataset(name: str, root: Path, **kwargs: object) -> Dataset:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(DATASETS)}")
    return DATASETS[name](root, **kwargs)  # type: ignore[arg-type]


__all__ = ["AMOS22", "CaseRecord", "DATASETS", "Dataset", "KiTS23", "get_dataset"]
