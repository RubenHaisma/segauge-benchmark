"""End-to-end: run_benchmark over a fake dataset + fake models, then render."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import segbench.datasets as datasets_mod
import segbench.models as models_mod
from conftest import cube, save_label
from segbench.datasets.base import CaseRecord, Dataset
from segbench.models.base import ModelAdapter
from segbench.run import RunConfig, run_benchmark
from segbench.schema import KITS23_SCHEMA
from segbench.site import render_index


class _FakeDataset(Dataset):
    name = "fake"
    schema = KITS23_SCHEMA
    license = "CC-BY-4.0"
    citation = "fake et al."

    def ensure_available(self, n=None):
        for i in range(n or 3):
            d = self.root / f"case_{i}"
            d.mkdir(parents=True, exist_ok=True)
            save_label(d / "image.nii.gz", np.zeros((40, 40, 40)))
            save_label(d / "gt.nii.gz", cube(label=1))

    def cases(self, n=None):
        out = []
        for i in range(n or 3):
            d = self.root / f"case_{i}"
            out.append(
                CaseRecord(f"case_{i}", d / "image.nii.gz", d / "gt.nii.gz",
                           {"modality": "CT"})
            )
        return out


class _FakeModel(ModelAdapter):
    license = "Apache-2.0"
    supports = frozenset({"kidney"})

    def __init__(self, shift: int = 0, name: str = "fake"):
        self.shift = shift
        self.name = name

    def organ_masks(self, image, out_dir, organs):
        m = np.zeros((40, 40, 40), dtype=bool)
        s = self.shift
        m[10 + s : 30 + s, 10:30, 10:30] = True
        return {"kidney": m}


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setitem(datasets_mod.DATASETS, "fake", _FakeDataset)
    monkeypatch.setitem(models_mod.MODELS, "fake", _FakeModel)


def test_end_to_end_run_and_render(registered, tmp_path: Path):
    cfg = RunConfig(
        dataset={"name": "fake", "root": str(tmp_path / "data")},
        models=[
            {"type": "fake", "shift": 0, "name": "good"},
            {"type": "fake", "shift": 6, "name": "bad"},
        ],
        n_cases=4,
        organs=["kidney"],
        n_resamples=200,
        nsd_tolerance=2.0,
        cache_dir=str(tmp_path / "preds"),
        results_path=str(tmp_path / "results.json"),
        preview=True,
        preview_label="test",
    )
    results = run_benchmark(cfg, log=lambda *_: None)

    assert results["dataset"]["n_cases"] == 4
    names = [m["name"] for m in results["models"]]
    assert names == ["good", "bad"]

    # The better (unshifted) model wins and the ranking reflects it.
    an = results["analysis"]["kidney"]["dice"]
    assert an["ranking"]["stats"][0]["name"] == "good"
    assert an["pairwise"][0]["favored"] == "good"

    # Render produces markdown that names both models and the dataset.
    md = render_index(results)
    assert "good" in md and "bad" in md
    assert "Kidney" in md
    assert "Ranking stability" in md
    # JSON written and reloadable.
    assert Path(cfg.results_path).exists()


def test_suspect_model_excluded_from_ranking(registered, tmp_path: Path):
    cfg = RunConfig(
        dataset={"name": "fake", "root": str(tmp_path / "data")},
        models=[
            {"type": "fake", "shift": 0, "name": "good"},
            # shift 60 pushes the cube entirely out of frame -> empty -> dice 0.
            {"type": "fake", "shift": 60, "name": "broken"},
        ],
        n_cases=3,
        organs=["kidney"],
        n_resamples=100,
        cache_dir=str(tmp_path / "preds"),
        results_path=str(tmp_path / "results.json"),
    )
    results = run_benchmark(cfg, log=lambda *_: None)
    broken = next(m for m in results["models"] if m["name"] == "broken")
    assert broken["per_organ"]["kidney"]["suspect"] is True
    # Excluded from ranking: only the good model remains rankable.
    stats = results["analysis"]["kidney"]["dice"]["ranking"]["stats"]
    assert [s["name"] for s in stats] == ["good"]
