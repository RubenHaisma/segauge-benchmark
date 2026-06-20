"""Model adapter remap + fairness contract."""

from __future__ import annotations

import numpy as np
import segauge as sg

from segbench.models import TotalSegmentator
from segbench.models.base import ModelAdapter
from segbench.schema import AMOS_CT_SCHEMA, KITS23_SCHEMA


class _FakeModel(ModelAdapter):
    name = "fake"
    license = "X"
    supports = frozenset({"kidney"})

    def __init__(self, shift: int = 0):
        self.shift = shift

    def organ_masks(self, image, out_dir, organs):
        m = np.zeros((40, 40, 40), dtype=bool)
        s = self.shift
        m[10 + s : 30 + s, 10:30, 10:30] = True
        return {"kidney": m}


def test_predict_remaps_to_schema_and_scores(kidney_case, tmp_path):
    image, gt = kidney_case
    pred = _FakeModel(shift=0).predict(image, gt, tmp_path / "out", KITS23_SCHEMA)
    res = sg.evaluate(
        [sg.Case("c", pred=str(pred), gt=str(gt))], label=1, detection=False
    )
    # Perfect overlap remapped to kidney id 1 -> dice ~ 1.
    assert res.summary()["dice"].value > 0.99


def test_shift_lowers_dice(kidney_case, tmp_path):
    image, gt = kidney_case
    p0 = _FakeModel(shift=0).predict(image, gt, tmp_path / "a", KITS23_SCHEMA)
    p5 = _FakeModel(shift=5).predict(image, gt, tmp_path / "b", KITS23_SCHEMA)
    d0 = sg.evaluate([sg.Case("c", pred=str(p0), gt=str(gt))], label=1)
    d5 = sg.evaluate([sg.Case("c", pred=str(p5), gt=str(gt))], label=1)
    assert d0.summary()["dice"].value > d5.summary()["dice"].value


def test_supported_in_intersects_schema():
    fm = _FakeModel()
    assert fm.supported_in(KITS23_SCHEMA) == ["kidney"]
    # AMOS has no plain "kidney" organ (it splits left/right), so a kidney-only
    # model supports none of AMOS's organs under name matching.
    assert fm.supported_in(AMOS_CT_SCHEMA) == []


def test_totalsegmentator_contamination():
    ts = TotalSegmentator(fast=True)
    assert ts.is_fair_on("kits23") is True
    assert ts.is_fair_on("totalsegmentator") is False
    assert "kidney" in ts.supported_in(KITS23_SCHEMA)
