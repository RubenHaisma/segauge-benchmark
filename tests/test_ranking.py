"""Ranking-stability + pairwise-significance behaviour (the rigor centerpiece)."""

from __future__ import annotations

import numpy as np
from segauge import paired_significance, ranking_stability

from segbench.ranking import analyze


def test_clear_separation_is_stable():
    rng = np.random.default_rng(0)
    a = np.clip(rng.normal(0.90, 0.02, 40), 0, 1)
    b = np.clip(rng.normal(0.70, 0.05, 40), 0, 1)
    r = ranking_stability({"A": a, "B": b}, higher_is_better=True, metric="dice")
    assert [s.name for s in r.stats] == ["A", "B"]
    assert r.stats[0].p_best == 1.0
    assert r.stats[0].mean_rank == 1.0


def test_near_tie_is_unstable():
    rng = np.random.default_rng(1)
    base = np.clip(rng.normal(0.85, 0.05, 30), 0, 1)
    a = base + rng.normal(0, 0.002, 30)
    b = base + rng.normal(0, 0.002, 30)
    r = ranking_stability({"A": a, "B": b}, higher_is_better=True)
    # Neither model owns rank 1; both p_best are strictly between 0 and 1.
    assert 0.0 < r.stats[0].p_best < 1.0


def test_lower_is_better_flips_order():
    # HD95-style: smaller is better.
    r = ranking_stability(
        {"good": [1.0, 1.2, 0.8], "bad": [5.0, 6.0, 4.0]},
        higher_is_better=False,
    )
    assert r.stats[0].name == "good"


def test_paired_significance_distinguishable_and_not():
    rng = np.random.default_rng(2)
    a = np.clip(rng.normal(0.90, 0.02, 40), 0, 1)
    b = np.clip(rng.normal(0.70, 0.03, 40), 0, 1)
    sep = paired_significance(a, b, a_name="A", b_name="B")
    assert sep.distinguishable and sep.favored == "A"

    tie = paired_significance(a, a + rng.normal(0, 1e-4, 40), a_name="A", b_name="A2")
    assert not tie.distinguishable and tie.favored is None


def test_analyze_produces_ranking_and_pairwise():
    scores = {
        "A": {"kidney": {"dice": {f"c{i}": 0.9 for i in range(10)}}},
        "B": {"kidney": {"dice": {f"c{i}": 0.7 for i in range(10)}}},
    }
    out = analyze(scores, ["kidney"], ["dice"], n_resamples=200, seed=0)
    an = out["kidney"]["dice"]
    assert an["n_cases"] == 10
    assert an["ranking"]["stats"][0]["name"] == "A"
    assert len(an["pairwise"]) == 1
    assert an["pairwise"][0]["favored"] == "A"


def test_analyze_aligns_on_common_cases_only():
    # B is missing one case; ranking must use the intersection.
    scores = {
        "A": {"kidney": {"dice": {"c0": 0.9, "c1": 0.9, "c2": 0.9}}},
        "B": {"kidney": {"dice": {"c0": 0.7, "c1": 0.7}}},
    }
    out = analyze(scores, ["kidney"], ["dice"], n_resamples=100, seed=0)
    assert out["kidney"]["dice"]["n_cases"] == 2
