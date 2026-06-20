"""Leaderboard analysis: turn per-case scores into rankings + significance.

This is the glue between the runner's per-case metric values and segauge's
statistics. For each (organ, metric) it ranks the models with a paired
case-resampling bootstrap and runs a pairwise significance test between the top
models, so the leaderboard can state not just an order but whether that order is
robust and whether the leader is actually separable from the runner-up.
"""

from __future__ import annotations

from segauge import paired_significance, ranking_stability
from segauge.types import PairedComparison, RankingResult

from segbench.schema import metric_higher_is_better

# {model: {organ: {metric: {case_id: value}}}}
ScoreTree = dict[str, dict[str, dict[str, dict[str, float]]]]


def _aligned(
    scores: ScoreTree, models: list[str], organ: str, metric: str
) -> tuple[list[str], dict[str, list[float]]]:
    """Common case ids (present for every model) and per-model aligned values."""
    per_model_cases = []
    for m in models:
        cases = scores.get(m, {}).get(organ, {}).get(metric, {})
        per_model_cases.append(set(cases))
    if not per_model_cases:
        return [], {}
    common = sorted(set.intersection(*per_model_cases)) if per_model_cases else []
    aligned = {
        m: [scores[m][organ][metric][c] for c in common] for m in models
    }
    return common, aligned


def _ranking_to_dict(r: RankingResult) -> dict[str, object]:
    return {
        "metric": r.metric,
        "higher_is_better": r.higher_is_better,
        "n_cases": r.n_cases,
        "n_resamples": r.n_resamples,
        "stats": [
            {
                "name": s.name,
                "score": s.score,
                "p_best": s.p_best,
                "mean_rank": s.mean_rank,
                "rank_ci_low": s.rank_ci_low,
                "rank_ci_high": s.rank_ci_high,
            }
            for s in r.stats
        ],
    }


def _pair_to_dict(p: PairedComparison) -> dict[str, object]:
    return {
        "a": p.a,
        "b": p.b,
        "delta": p.delta,
        "ci_low": p.ci_low,
        "ci_high": p.ci_high,
        "favored": p.favored,
        "distinguishable": p.distinguishable,
    }


def analyze(
    scores: ScoreTree,
    organs: list[str],
    metrics: list[str],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, object]]]:
    """Produce {organ: {metric: {ranking, pairwise}}} over all eligible models."""
    out: dict[str, dict[str, dict[str, object]]] = {}
    for organ in organs:
        out[organ] = {}
        for metric in metrics:
            # Models that actually have scores for this organ+metric.
            models = [
                m
                for m in scores
                if scores[m].get(organ, {}).get(metric)
            ]
            if len(models) < 1:
                continue
            common, aligned = _aligned(scores, models, organ, metric)
            if not common:
                continue
            higher = metric_higher_is_better(metric)
            ranking = ranking_stability(
                aligned,
                higher_is_better=higher,
                n_resamples=n_resamples,
                seed=seed,
                metric=metric,
            )
            # Pairwise significance between every pair, ordered best-first.
            ordered = [s.name for s in ranking.stats]
            pairwise = []
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    a, b = ordered[i], ordered[j]
                    pairwise.append(
                        _pair_to_dict(
                            paired_significance(
                                aligned[a],
                                aligned[b],
                                a_name=a,
                                b_name=b,
                                higher_is_better=higher,
                                n_resamples=n_resamples,
                                seed=seed,
                            )
                        )
                    )
            out[organ][metric] = {
                "n_cases": len(common),
                "ranking": _ranking_to_dict(ranking),
                "pairwise": pairwise,
            }
    return out
