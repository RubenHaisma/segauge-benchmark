"""Command-line interface: ``segbench run``, ``render``, and ``reanalyze``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from segbench.ranking import analyze, score_tree_from_results
from segbench.run import RunConfig, run_benchmark
from segbench.site import render_site


def _run(args: argparse.Namespace) -> int:
    cfg_dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = RunConfig.from_dict(cfg_dict)
    if args.n_cases is not None:
        cfg.n_cases = args.n_cases
    results = run_benchmark(cfg)
    if args.render:
        render_site(results, Path(args.render))
    return 0


def _render(args: argparse.Namespace) -> int:
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    render_site(results, Path(args.out))
    return 0


def _recompute_analysis(results: dict) -> dict:
    """Re-derive the ranking analysis purely from the published per-case scores."""
    cfg = results.get("config", {})
    organs = [o["name"] for o in results["dataset"]["organs"]]
    tree = score_tree_from_results(results)
    return analyze(
        tree,
        organs,
        list(results.get("metrics", [])),
        n_resamples=cfg.get("n_resamples", 2000),
        seed=cfg.get("seed", 0),
        confidence=cfg.get("confidence", 0.95),
    )


def _reanalyze(args: argparse.Namespace) -> int:
    """Reproduce the ranking statistics from a results JSON, with no inference.

    The per-case scores in the JSON fully determine the ranking and significance
    verdicts, so this regenerates them deterministically. With no flags it
    *verifies* the committed analysis matches (exit 1 on drift) which is the CI
    reproducibility gate; ``--write`` persists, ``--render`` rebuilds the site.
    """
    path = Path(args.results)
    results = json.loads(path.read_text(encoding="utf-8"))
    recomputed = _recompute_analysis(results)

    stored = results.get("analysis")
    matches = stored == recomputed
    if not matches:
        print(
            f"[segbench] reanalyze: recomputed analysis differs from the "
            f"analysis stored in {path}"
        )

    if args.write:
        results["analysis"] = recomputed
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[segbench] wrote recomputed analysis to {path}")
    if args.render:
        results["analysis"] = recomputed
        render_site(results, Path(args.render))
        print(f"[segbench] rendered site to {args.render}")

    if args.write or args.render:
        return 0
    if matches:
        print(
            f"[segbench] reanalyze: OK - the published ranking statistics in "
            f"{path} reproduce exactly from its per-case scores"
        )
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="segbench",
        description="Contamination-aware, statistically-rigorous medical "
        "segmentation leaderboard.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the benchmark from a config")
    p_run.add_argument("--config", required=True, help="path to a run config YAML")
    p_run.add_argument("--n-cases", type=int, default=None, help="override n_cases")
    p_run.add_argument(
        "--render", default=None, help="also render the site to this docs dir"
    )
    p_run.set_defaults(func=_run)

    p_render = sub.add_parser("render", help="render the site from a results JSON")
    p_render.add_argument("--results", required=True, help="path to results JSON")
    p_render.add_argument("--out", required=True, help="output docs directory")
    p_render.set_defaults(func=_render)

    p_re = sub.add_parser(
        "reanalyze",
        help="reproduce (and verify) the ranking statistics from a results JSON",
    )
    p_re.add_argument("--results", required=True, help="path to results JSON")
    p_re.add_argument(
        "--write",
        action="store_true",
        help="persist the recomputed analysis back into the JSON",
    )
    p_re.add_argument(
        "--render", default=None, help="also render the site to this docs dir"
    )
    p_re.set_defaults(func=_reanalyze)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
