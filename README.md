# segauge-benchmark

**An independent, reproducible leaderboard for medical image segmentation models, where every score has a confidence interval and a ranking-stability test, and you can reproduce the whole thing with one command.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Powered by segauge](https://img.shields.io/badge/powered%20by-segauge-0a7.svg)](https://github.com/RubenHaisma/segauge)
[![Docs](https://img.shields.io/badge/leaderboard-live-success.svg)](https://rubenhaisma.github.io/segauge-benchmark/)

Most medical image segmentation leaderboards report a single Dice number per model with no interval, so you cannot tell a real lead from sampling noise. The people who study segmentation metrics have shown this is not a nitpick:

- Removing **a single test case** changes the rank of most teams in a **majority** of segmentation challenges (Maier-Hein et al., *Nature Communications* 2018).
- In **over 60%** of MICCAI 2023 segmentation papers, the second-ranked method sits **inside the first's confidence interval**, i.e. the reported "winner" is not statistically distinguishable (*Confidence Intervals Uncovered*, MICCAI 2024).

`segauge-benchmark` turns that critique into a running tool. It:

- runs real models on **public** CT/MRI data,
- scores them with [**segauge**](https://github.com/RubenHaisma/segauge) (Dice, **HD95**, **Normalized Surface Dice**, ASSD), putting a **bootstrap confidence interval** on every number,
- reports a **ranking-stability** test (how often each model is actually rank 1 under case resampling) and a **pairwise significance** test (is #1 separable from #2?),
- is **contamination-aware**: a model is only ranked on a dataset it was **not** trained on, and
- is **reproducible with one command** on data anyone can download.

## The honest positioning

There is already a good independent benchmark in this space, [**Touchstone**](https://github.com/MrGiovanni/Touchstone) (NeurIPS 2024), which evaluates models on large multi-center CT data. We do **not** claim better or secret data. The wedge here is different and complementary:

- **Statistics first.** Every row carries a CI and a ranking-stability result. A submission-server benchmark structurally cannot do this, because it never returns per-case predictions.
- **Reproducible on public data.** You can re-run every number yourself with `segbench run`. The trade-off, stated plainly, is that public-ground-truth data is in-distribution for some models; we handle that with an explicit contamination policy rather than by pretending it away.
- **Failure-mode slicing.** Per-organ today; per-scanner and demographic where the dataset provides it.

## Quickstart

```bash
git clone https://github.com/RubenHaisma/segauge-benchmark
cd segauge-benchmark
uv sync

# Reproduce the seed leaderboard (downloads a few KiTS23 cases, runs on CPU):
uv run segbench run --config configs/kits23_seed.yaml --render docs

# Render the site from an existing results file:
uv run segbench render --results results/kits23_seed.json --out docs
```

The live leaderboard: **https://rubenhaisma.github.io/segauge-benchmark/**

## Status

Early **preview**, but already real and multi-model. The current leaderboard compares **four genuinely different whole-body CT models** on whole-kidney segmentation over KiTS23 cases, run on GPU:

- **TotalSegmentator** (the de-facto nnU-Net-based reference)
- **MOOSE** (`clin_ct_organs`, from the nuclear-medicine / PET-CT world — an independent lineage)
- **MONAI wholeBody** (the framework's own bundle)
- **CT-FM** (a SegResNet foundation model)

At N=20 the three rendered models are **statistically indistinguishable** on kidney (every pairwise difference has a confidence interval through zero) — which is precisely the point: a bare Dice table would have declared a "winner". Next steps: the AMOS22 multi-organ run (15 organs, per-organ failure slicing) and adding STU-Net.

### Running on GPU (Modal)

Inference runs on [Modal](https://modal.com) serverless GPUs, one isolated image per model:

```bash
# Multi-model run on KiTS23 (downloads cases into a Modal Volume, runs each model on a GPU):
uv run modal run modal_app.py::main --config configs/kits23_modal.yaml --n-cases 20

# One-time AMOS22 download into the volume, then the multi-organ run:
uv run modal run modal_app.py::prep_amos
uv run modal run modal_app.py::main --config configs/amos_modal.yaml --n-cases 20
```

The harness also runs locally on CPU for a single model (`uv run segbench run --config configs/kits23_seed.yaml`).

## How a model and a dataset plug in

- A **dataset adapter** (`src/segbench/datasets/`) yields cases (image, ground-truth label map, metadata) and declares its label schema and license.
- A **model adapter** (`src/segbench/models/`) runs inference in its **own** environment and returns one mask per organ it supports, remapped into the dataset's label schema **by organ name**. The harness never imports torch.
- The **runner** (`src/segbench/run.py`) scores every supported organ with segauge, spot-checks each model so a misaligned adapter cannot silently publish zeros, and writes one results JSON that fully describes the run.

See [the methodology page](https://rubenhaisma.github.io/segauge-benchmark/methodology/) for the contamination policy, the statistics, and the per-model label mapping.

## Datasets and licensing

The leaderboard publishes derived **scores**, never re-hosted images. It stays non-commercial where a dataset's license (e.g. KiTS23, CC BY-NC-SA 4.0) requires it, and cites every dataset. The commercially-safe (CC-BY family) datasets used or planned are AMOS22, the TotalSegmentator dataset, MSD, and BTCV.

## License

Apache-2.0. Built on [segauge](https://github.com/RubenHaisma/segauge).
