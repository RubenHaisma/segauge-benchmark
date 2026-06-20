"""Command-line interface: ``segbench run`` and ``segbench render``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

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

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
