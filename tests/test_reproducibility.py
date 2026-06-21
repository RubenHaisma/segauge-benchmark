"""The published leaderboard must be reproducible from the published data alone.

These tests are the teeth behind "reproducible with one command": every committed
results JSON stores its per-case scores, and the headline ranking + significance
verdicts must follow *exactly* from those numbers, deterministically and with no
inference. If an edit ever changes the statistics without the published per-case
data changing, CI fails here. The rendered site must likewise be exactly what the
committed results produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from segbench.ranking import (
    _adjusted_confidence,
    analyze,
    score_tree_from_results,
)
from segbench.site import render_index

REPO = Path(__file__).resolve().parents[1]
RESULTS = sorted((REPO / "results").glob("*.json"))


def _recompute(results: dict) -> dict:
    cfg = results.get("config", {})
    organs = [o["name"] for o in results["dataset"]["organs"]]
    return analyze(
        score_tree_from_results(results),
        organs,
        list(results.get("metrics", [])),
        n_resamples=cfg.get("n_resamples", 2000),
        seed=cfg.get("seed", 0),
        confidence=cfg.get("confidence", 0.95),
    )


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.name)
def test_published_stats_reproduce_from_per_case_scores(path: Path):
    results = json.loads(path.read_text(encoding="utf-8"))
    assert _recompute(results) == results["analysis"], (
        f"{path.name}: the committed ranking analysis does not match what its "
        "own per-case scores produce; run `segbench reanalyze --results "
        f"{path} --write --render docs` and commit."
    )


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.name)
def test_analysis_is_deterministic(path: Path):
    results = json.loads(path.read_text(encoding="utf-8"))
    assert _recompute(results) == _recompute(results)


def test_committed_site_matches_results():
    """docs/index.md must be exactly what the modal results render to."""
    results = json.loads(
        (REPO / "results" / "kits23_modal.json").read_text(encoding="utf-8")
    )
    rendered = render_index(results)
    committed = (REPO / "docs" / "index.md").read_text(encoding="utf-8")
    assert rendered == committed, (
        "docs/index.md is stale; regenerate with "
        "`segbench render --results results/kits23_modal.json --out docs`."
    )


def test_render_is_deterministic():
    results = json.loads(
        (REPO / "results" / "kits23_modal.json").read_text(encoding="utf-8")
    )
    assert render_index(results) == render_index(results)


def test_score_tree_drops_suspect_and_missing():
    results = {
        "metrics": ["dice", "hd95"],
        "models": [
            {
                "name": "good",
                "per_organ": {
                    "kidney": {
                        "suspect": False,
                        "per_case": [
                            {"case_id": "c0", "dice": 0.9, "hd95": 3.0},
                            {"case_id": "c1", "dice": 0.8, "hd95": None},
                        ],
                    },
                    "spleen": {  # excluded: spot-check failed
                        "suspect": True,
                        "per_case": [{"case_id": "c0", "dice": 0.02, "hd95": 99.0}],
                    },
                },
            }
        ],
    }
    tree = score_tree_from_results(results)
    assert set(tree["good"]) == {"kidney"}  # suspect spleen dropped
    assert tree["good"]["kidney"]["dice"] == {"c0": 0.9, "c1": 0.8}
    assert tree["good"]["kidney"]["hd95"] == {"c0": 3.0}  # None dropped


def test_bonferroni_widens_only_with_multiple_comparisons():
    assert _adjusted_confidence(0.95, 0, "bonferroni") == 0.95
    assert _adjusted_confidence(0.95, 1, "bonferroni") == 0.95
    # Three comparisons: 1 - 0.05/3.
    assert _adjusted_confidence(0.95, 3, "bonferroni") == pytest.approx(1 - 0.05 / 3)
    assert _adjusted_confidence(0.95, 3, "none") == 0.95
