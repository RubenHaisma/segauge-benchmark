"""Render the results JSON into the public leaderboard page (Markdown for mkdocs).

The generated ``index.md`` leads with the finding that makes the project worth
linking: not just an order, but whether that order is statistically real. Every
cell is a value with its confidence interval; every organ gets a ranking-
stability line and a pairwise-significance verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

UNIT = {"hd95": " mm", "assd": " mm", "dice": "", "nsd": "", "iou": ""}
PRETTY = {"dice": "Dice", "hd95": "HD95", "nsd": "NSD", "assd": "ASSD", "iou": "IoU"}


def _fmt_est(est: dict[str, Any] | None, unit: str = "") -> str:
    if not est or est.get("value") is None:
        return "n/a"
    v = est["value"]
    lo, hi = est.get("ci_low"), est.get("ci_high")
    if lo is None or hi is None:
        return f"{v:.3g}{unit}"
    return f"{v:.3g} [{lo:.3g}, {hi:.3g}]{unit}"


def _model_by_name(results: dict[str, Any], name: str) -> dict[str, Any] | None:
    for m in results["models"]:
        if m["name"] == name:
            return m
    return None


def _leaderboard_tables(results: dict[str, Any]) -> list[str]:
    metrics = results["metrics"]
    organs = [o["name"] for o in results["dataset"]["organs"]]
    analysis = results.get("analysis", {})
    lines: list[str] = []

    for organ in organs:
        # Which models reported this organ?
        rows = []
        for m in results["models"]:
            cell = m.get("per_organ", {}).get(organ)
            if cell is None:
                continue
            rows.append((m, cell))
        if not rows:
            continue

        lines.append(f"## {organ.capitalize()}\n")

        header = "| Model | Fair | " + " | ".join(
            f"{PRETTY.get(x, x)}{(' (mm)' if UNIT.get(x) else '')}" for x in metrics
        ) + " |"
        sep = "|" + "---|" * (len(metrics) + 2)
        lines.append(header)
        lines.append(sep)
        for m, cell in rows:
            fair = "yes" if m.get("fair", True) else "no (trained on this)"
            suspect = " ⚠️" if cell.get("suspect") else ""
            cols = [_fmt_est(cell["summary"].get(x)) for x in metrics]
            row = f"| {m['name']}{suspect} | {fair} | " + " | ".join(cols) + " |"
            lines.append(row)
        lines.append("")

        # Ranking stability + significance, by the primary metric (dice).
        organ_an = analysis.get(organ, {})
        primary = "dice" if "dice" in organ_an else next(iter(organ_an), None)
        if primary and organ_an.get(primary):
            an = organ_an[primary]
            n = an.get("n_cases")
            lines.append(
                f"**Ranking stability** (by {PRETTY.get(primary, primary)}, "
                f"paired bootstrap over n={n} cases):\n"
            )
            for s in an["ranking"]["stats"]:
                lines.append(
                    f"- {s['name']}: {s['score']:.3g} "
                    f"(P(rank 1) = {s['p_best']:.0%}, "
                    f"mean rank {s['mean_rank']:.1f} "
                    f"[{s['rank_ci_low']:.0f}, {s['rank_ci_high']:.0f}])"
                )
            lines.append("")
            verdicts = _significance_lines(an["pairwise"])
            if verdicts:
                lines.append("**Statistical separability:**\n")
                lines.extend(verdicts)
                lines.append("")
    return lines


def _significance_lines(pairwise: list[dict[str, Any]]) -> list[str]:
    out = []
    for p in pairwise:
        if p["distinguishable"]:
            loser = p["b"] if p["favored"] == p["a"] else p["a"]
            out.append(
                f"- **{p['favored']}** beats {loser} "
                f"(Δ={p['delta']:.3g} [{p['ci_low']:.3g}, {p['ci_high']:.3g}])"
            )
        else:
            out.append(
                f"- **{p['a']} and {p['b']} are not statistically separable** "
                f"(Δ={p['delta']:.3g}, CI includes 0)"
            )
    return out


def render_index(results: dict[str, Any]) -> str:
    ds = results["dataset"]
    preview = results.get("preview")
    banner = (
        f"!!! warning \"{results.get('preview_label') or 'Preview run'}\"\n"
        "    This is an early, small-N preview that demonstrates the methodology "
        "end to end on real, public data. The wide intervals are the point: they "
        "show how little a bare leaderboard number tells you at this sample size. "
        "Scaling to more models and organs is the next step.\n\n"
        if preview
        else ""
    )

    cfg = results["config"]
    conf_pct = int(cfg["confidence"] * 100)
    seg_v = results["segauge_version"]
    ds_line = (
        f"**Dataset:** {ds['name']} ({ds['n_cases']} cases, "
        f"ground-truth labels {ds['license']}). "
        f"Scored with [segauge](https://github.com/RubenHaisma/segauge) v{seg_v} "
        f"({conf_pct}% bootstrap CI, {cfg['n_resamples']} resamples, "
        f"seed {cfg['seed']})."
    )

    head = f"""# The segauge medical segmentation leaderboard

**Every score with a confidence interval and a ranking-stability test. Sliced by
failure mode. On public data. Reproducible with one command.**

Most segmentation leaderboards report a single Dice number with no interval, so
you cannot tell a real lead from sampling noise. The medical-imaging metrics
community has shown this repeatedly: removing one test case flips most teams'
ranks in a majority of challenges, and the reported "winner" often sits inside
the runner-up's confidence interval. This leaderboard is that critique turned
into a running tool.

{banner}{ds_line}
See [Methodology](methodology.md) and [Reproduce](reproduce.md).

A leaderboard you can run yourself: `pip install segauge` and one `segbench run`.

"""
    body = "\n".join(_leaderboard_tables(results))
    foot = (
        "\n---\n\n"
        "*Contamination policy:* a model is only ranked on a dataset it was not "
        "trained on. Cells marked \"no\" are shown for context but excluded from "
        "ranking. *Dataset citation:* " + ds.get("citation", "") + "\n"
    )
    return head + body + foot


def render_site(results: dict[str, Any], docs_dir: Path) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    index = docs_dir / "index.md"
    index.write_text(render_index(results), encoding="utf-8")
    return index
