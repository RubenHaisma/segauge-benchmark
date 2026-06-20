"""segauge-benchmark — an independent, contamination-aware, statistically-rigorous
public leaderboard for medical image segmentation models.

Every score carries a bootstrap confidence interval and a ranking-stability test,
is sliced by failure mode, and is reproducible with one command on public data.
Powered by segauge (https://github.com/RubenHaisma/segauge).
"""

from __future__ import annotations

from segbench.ranking import analyze
from segbench.run import RunConfig, run_benchmark
from segbench.schema import LEADERBOARD_METRICS, LabelSchema, Organ

__version__ = "0.1.0"

__all__ = [
    "LEADERBOARD_METRICS",
    "LabelSchema",
    "Organ",
    "RunConfig",
    "__version__",
    "analyze",
    "run_benchmark",
]
