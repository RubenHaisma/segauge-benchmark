# Reproduce

Every score on this leaderboard is reproducible from public data. Nothing here
depends on private weights or hidden test sets.

## Prerequisites

```bash
git clone https://github.com/RubenHaisma/segauge-benchmark
cd segauge-benchmark
uv sync
```

## Run the seed leaderboard

The seed run downloads a few KiTS23 cases (per case, a few hundred MB each) and
scores kidney segmentation on CPU:

```bash
uv run segbench run --config configs/kits23_seed.yaml --render docs
```

This writes `results/kits23_seed.json` and regenerates `docs/index.md`. The
results JSON fully describes the run (dataset, models, per-case scores, CIs,
ranking analysis), so the site can be regenerated without re-running inference:

```bash
uv run segbench render --results results/kits23_seed.json --out docs
```

## Model environments

Model inference runs in **isolated** environments so the harness stays light and
each model can be pinned independently. Point the adapters at those environments
with environment variables:

```bash
# TotalSegmentator (its own venv, with `pip install TotalSegmentator`):
export SEGBENCH_TS_BIN=/path/to/ts-venv/bin/TotalSegmentator

# CT-FM (a venv with `lighter_zoo` + `monai`):
export SEGBENCH_CTFM_PY=/path/to/ctfm-venv/bin/python
```

A model whose environment is not provisioned is skipped with a warning; the run
still completes with the models that are available.

## Adding a model or dataset

- A dataset adapter lives in `src/segbench/datasets/` and yields cases plus a
  label schema and licence.
- A model adapter lives in `src/segbench/models/` and returns one mask per organ
  it supports, remapped into the dataset schema by organ name.

Both plug into the config; see `configs/kits23_seed.yaml`.
