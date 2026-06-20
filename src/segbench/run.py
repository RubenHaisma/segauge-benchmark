"""The benchmark runner.

Config-driven, deterministic, and resumable: model predictions are cached on
disk, so re-running only re-scores. It downloads the chosen cases, runs each
model, scores every supported organ with segauge (per-case metrics + bootstrap
CIs), guards against silently-broken adapters with a per-model spot-check, runs
the ranking-stability analysis, and writes one results JSON that fully describes
the run. That JSON is the only large-ish artifact committed to the repo and is
what the site renderer consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import segauge as sg

from segbench.datasets import get_dataset
from segbench.models import build_model
from segbench.ranking import analyze
from segbench.schema import LEADERBOARD_METRICS


@dataclass
class RunConfig:
    dataset: dict[str, Any]
    models: list[dict[str, Any]]
    n_cases: int = 5
    organs: list[str] | None = None
    metrics: tuple[str, ...] = LEADERBOARD_METRICS
    confidence: float = 0.95
    n_resamples: int = 2000
    seed: int = 0
    nsd_tolerance: float = 2.0
    spot_check_dice: float = 0.1
    preview: bool = True
    preview_label: str = ""
    cache_dir: str = "predictions"
    results_path: str = "results/results.json"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _sanitize(obj: Any) -> Any:
    """Make a structure JSON-safe: non-finite floats become None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _est(e: sg.Estimate) -> dict[str, float]:
    return {"value": e.value, "ci_low": e.ci_low, "ci_high": e.ci_high}


# {organ_name: {metric: {case_id: value}}}
ModelScores = dict[str, dict[str, dict[str, float]]]


def _prepare(cfg: RunConfig, *, log=print):
    """Build the dataset, download the cases, and resolve requested organs."""
    ds_spec = dict(cfg.dataset)
    ds_name = ds_spec.pop("name")
    ds_root = Path(ds_spec.pop("root"))
    dataset = get_dataset(ds_name, ds_root, **ds_spec)
    schema = dataset.schema
    requested = (
        [o for o in schema.organs if o.name in cfg.organs]
        if cfg.organs
        else list(schema.organs)
    )
    log(f"[segbench] downloading {cfg.n_cases} case(s) of {ds_name} ...")
    dataset.ensure_available(cfg.n_cases)
    cases = dataset.cases(cfg.n_cases)
    log(f"[segbench] {len(cases)} case(s) ready")
    return dataset, schema, requested, cases, ds_name


def evaluate_model(model, cases, requested, schema, ds_name, cfg, *, log=print):
    """Run + score ONE model. Returns (model_entry, scores) or (None, {}).

    Scores feed ranking; only organs that pass the spot-check are included.
    Inference is cached on disk, so this is cheap to re-run.
    """
    supported_keys = set(model.supported_in(schema))
    organs = [o for o in requested if o.key in supported_keys]
    if not organs:
        log(f"[segbench] {model.name}: supports none of the requested organs")
        return None, {}
    log(f"[segbench] {model.name}: organs {[o.name for o in organs]}")

    cache_root = Path(cfg.cache_dir)
    # Per-case tolerance: a single case that fails (e.g. an out-of-memory case)
    # is skipped, not allowed to drop the whole model. Ranking later aligns on
    # the case ids common to every model.
    case_preds: list[tuple[str, Path, Path, dict[str, Any]]] = []
    for case in cases:
        out_dir = cache_root / model.name / case.case_id
        try:
            pred = model.predict(case.image, case.gt, out_dir, schema)
        except Exception as exc:  # noqa: BLE001 - report and continue
            log(f"[segbench] {model.name}/{case.case_id}: failed ({exc}); skip case")
            continue
        case_preds.append((case.case_id, pred, case.gt, dict(case.metadata)))
    if not case_preds:
        log(f"[segbench] {model.name}: no cases succeeded; skipping model")
        return None, {}

    per_organ: dict[str, Any] = {}
    model_scores: ModelScores = {}
    for organ in organs:
        seg_cases = [
            sg.Case(cid, pred=str(pred), gt=str(gt), metadata=meta)
            for (cid, pred, gt, meta) in case_preds
        ]
        res = sg.evaluate(
            seg_cases,
            label=organ.id,
            detection=False,
            nsd_tolerance=cfg.nsd_tolerance,
            confidence=cfg.confidence,
            n_resamples=cfg.n_resamples,
            seed=cfg.seed,
        )
        summary = res.summary()
        per_case = [
            {"case_id": r.get("case_id"), **{m: r.get(m) for m in cfg.metrics}}
            for r in res.rows
        ]
        mean_dice = summary["dice"].value
        suspect = not (mean_dice >= cfg.spot_check_dice)
        per_organ[organ.name] = {
            "summary": {m: _est(summary[m]) for m in cfg.metrics},
            "per_case": per_case,
            "suspect": suspect,
        }
        if suspect:
            log(
                f"[segbench] WARNING {model.name}/{organ.name}: mean dice "
                f"{mean_dice:.3f} < {cfg.spot_check_dice} -> excluded from "
                "ranking (likely label/orientation mismatch)"
            )
            continue
        model_scores[organ.name] = {
            m: {
                str(r.get("case_id")): float(r[m])
                for r in res.rows
                if r.get(m) is not None
            }
            for m in cfg.metrics
        }

    entry = {
        "name": model.name,
        "license": model.license,
        "version": model.version(),
        "contaminated_on": sorted(model.contaminated_on),
        "fair": model.is_fair_on(ds_name),
        "per_organ": per_organ,
    }
    return entry, model_scores


def assemble_results(
    dataset, n_cases, model_entries, scores, organ_names, cfg
) -> dict[str, Any]:
    """Merge per-model results into the final JSON with the ranking analysis."""
    analysis = analyze(
        scores, organ_names, list(cfg.metrics),
        n_resamples=cfg.n_resamples, seed=cfg.seed,
    )
    results = {
        "benchmark": "segauge-benchmark",
        "segauge_version": sg.__version__,
        "preview": cfg.preview,
        "preview_label": cfg.preview_label,
        "dataset": {**dataset.describe(), "n_cases": n_cases},
        "metrics": list(cfg.metrics),
        "config": {
            "confidence": cfg.confidence,
            "n_resamples": cfg.n_resamples,
            "seed": cfg.seed,
            "nsd_tolerance": cfg.nsd_tolerance,
            "spot_check_dice": cfg.spot_check_dice,
        },
        "models": model_entries,
        "analysis": analysis,
    }
    return _sanitize(results)


def run_single_model(cfg: RunConfig, model_spec: dict[str, Any], *, log=print):
    """Prepare the dataset and run+score exactly one model. For remote workers."""
    dataset, schema, requested, cases, ds_name = _prepare(cfg, log=log)
    model = build_model(model_spec)
    return evaluate_model(model, cases, requested, schema, ds_name, cfg, log=log)


def run_benchmark(cfg: RunConfig, *, log=print) -> dict[str, Any]:
    dataset, schema, requested, cases, ds_name = _prepare(cfg, log=log)

    scores: dict[str, ModelScores] = {}
    model_entries: list[dict[str, Any]] = []
    for spec in cfg.models:
        model = build_model(spec)
        entry, msc = evaluate_model(
            model, cases, requested, schema, ds_name, cfg, log=log
        )
        if entry is None:
            continue
        model_entries.append(entry)
        if msc:
            scores[entry["name"]] = msc

    organ_names = [o.name for o in requested]
    results = assemble_results(
        dataset, len(cases), model_entries, scores, organ_names, cfg
    )

    out_path = Path(cfg.results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"[segbench] wrote {out_path}")
    return results
