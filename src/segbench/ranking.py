"""Leaderboard analysis: turn per-case scores into rankings + significance.

This is the glue between the runner's per-case metric values and segauge's
statistics. For each (organ, metric) it ranks the models with a paired
case-resampling bootstrap and runs a pairwise significance test between the top
models, so the leaderboard can state not just an order but whether that order is
robust and whether the leader is actually separable from the runner-up.

Reporting many pairwise comparisons inflates the chance that one looks
"separable" by accident, so the pairwise intervals are widened with a
family-wise (Bonferroni) correction over the comparisons made for each
(organ, metric). See ``analyze``.
"""

from __future__ import annotations

from typing import Any

from segauge import paired_significance, ranking_stability
from segauge.types import PairedComparison, RankingResult

from segbench.schema import metric_higher_is_better

# {model: {organ: {metric: {case_id: value}}}}
ScoreTree = dict[str, dict[str, dict[str, dict[str, float]]]]


def _adjusted_confidence(
    confidence: float, n_comparisons: int, correction: str
) -> float:
    """Family-wise confidence level for one of ``n_comparisons`` pairwise tests.

    Bonferroni: to hold a family-wise error of ``1 - confidence`` across
    ``n_comparisons`` simultaneous comparisons, each individual interval is
    computed at ``1 - alpha / n_comparisons``. With one (or zero) comparison
    there is no family, so the level is unchanged.
    """
    if correction == "bonferroni" and n_comparisons > 1:
        alpha = 1.0 - confidence
        return 1.0 - alpha / n_comparisons
    return confidence


def score_tree_from_results(results: dict[str, Any]) -> ScoreTree:
    """Rebuild the per-case score tree from a results JSON.

    The published results JSON stores every model's per-case metric values, so
    the entire ranking analysis can be regenerated from it with no inference and
    no GPU. This is what makes the headline statistics independently verifiable:
    they must follow exactly from the per-case numbers on the page. Mirrors the
    runner's ranking semantics: organs flagged ``suspect`` are excluded, and a
    missing (``None``) metric value for a case is dropped.
    """
    metrics = results.get("metrics", [])
    tree: ScoreTree = {}
    for model in results.get("models", []):
        name = model["name"]
        per_organ = model.get("per_organ", {})
        organ_scores: dict[str, dict[str, dict[str, float]]] = {}
        for organ, cell in per_organ.items():
            if cell.get("suspect"):
                continue
            rows = cell.get("per_case", [])
            metric_scores: dict[str, dict[str, float]] = {}
            for metric in metrics:
                vals = {
                    str(r.get("case_id")): float(r[metric])
                    for r in rows
                    if r.get(metric) is not None
                }
                if vals:
                    metric_scores[metric] = vals
            if metric_scores:
                organ_scores[organ] = metric_scores
        if organ_scores:
            tree[name] = organ_scores
    return tree


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
    confidence: float = 0.95,
    correction: str = "bonferroni",
) -> dict[str, dict[str, dict[str, object]]]:
    """Produce {organ: {metric: {ranking, pairwise}}} over all eligible models.

    Pairwise significance intervals are computed at a family-wise corrected
    confidence level (``correction``, default Bonferroni) over the comparisons
    made for that (organ, metric), so reporting many pairs does not manufacture a
    spurious "separable" verdict.
    """
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
            n_comparisons = len(ordered) * (len(ordered) - 1) // 2
            adjusted = _adjusted_confidence(confidence, n_comparisons, correction)
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
                                confidence=adjusted,
                                n_resamples=n_resamples,
                                seed=seed,
                            )
                        )
                    )
            out[organ][metric] = {
                "n_cases": len(common),
                "confidence": confidence,
                "n_comparisons": n_comparisons,
                "correction": correction if n_comparisons > 1 else "none",
                "adjusted_confidence": adjusted,
                "ranking": _ranking_to_dict(ranking),
                "pairwise": pairwise,
            }
    return out
