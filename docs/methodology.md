# Methodology

This leaderboard is built to be defensible. Every choice below exists to make a
score mean what it says.

## Metrics

All metrics are computed by [segauge](https://github.com/RubenHaisma/segauge),
which evaluates 3D segmentations directly from NIfTI / DICOM:

- **Dice** and **IoU** — overlap.
- **HD95** — the 95th-percentile Hausdorff distance, in millimetres, computed on
  a surface mesh at true voxel spacing (robust to single-voxel outliers).
- **Normalized Surface Dice (NSD)** — the fraction of the surface within a
  clinical tolerance; the metric that tracks "is the boundary close enough".
- **ASSD** — average symmetric surface distance, in millimetres.

Every aggregate carries a **percentile bootstrap confidence interval** (default
95%, 2000 resamples, fixed seed), so the same inputs always give the same
interval.

## Why confidence intervals and ranking stability

A single Dice number hides how much it would move if the test set had been
slightly different. Two results from the medical-imaging metrics literature
motivate this whole project:

- Maier-Hein et al. (*Nature Communications*, 2018) found that removing **one**
  test case changed the rank of most teams in a **majority** of segmentation
  challenges.
- *Confidence Intervals Uncovered* (MICCAI 2024) found that in **over 60%** of
  MICCAI 2023 segmentation papers, the second method sits **inside** the first's
  confidence interval.

So for every (organ, metric) we report:

- **Ranking stability** — a paired bootstrap over the shared cases (the *same*
  resampled case indices are applied to every model, because they were scored on
  the same cases). We report each model's probability of being rank 1, its mean
  rank, and a rank interval.
- **Pairwise significance** — the bootstrap CI of the mean per-case difference
  between two models. If that interval includes zero, the two are reported as
  **not statistically separable**, however different their point scores look.

## Contamination policy

The honest weakness of any benchmark built on **public** ground truth is that
some models trained on that data, so their score there is in-distribution and
not a fair test. We do not paper over this:

- Each model declares the datasets it was trained on.
- A (model, dataset) cell where the model trained on the dataset is marked
  **not fair** and **excluded from ranking**. It may still be shown for context.
- We never claim to have private or multi-center hidden test data. Where that
  matters, [Touchstone](https://github.com/MrGiovanni/Touchstone) is the
  reference; this project is the reproducible, statistics-first complement.

## Label mapping

Models emit different label numberings. We map each model's output into the
dataset's schema **by organ name**, then score per organ. A wrong mapping would
silently score the wrong structure, so:

- mappings are published per model (in the adapter source), and
- the runner **spot-checks** every model on every organ: if the mean Dice is
  below a floor (default 0.1) the cell is flagged and excluded, which catches an
  orientation or label-id mismatch instead of publishing a misleading zero or a
  wrong overlap.

A model is only scored on organs it actually supports. A general organ model
that cannot segment kidney tumors is shown **n/a** for tumor, never a silent
zero.

## Datasets and licensing

The leaderboard publishes derived **scores**, never re-hosted images. It honours
each dataset's licence (staying non-commercial for e.g. KiTS23, CC BY-NC-SA 4.0)
and cites it. The commercially-safe (CC-BY family) datasets used or planned are
AMOS22, the TotalSegmentator dataset, MSD, and BTCV.

## Reproducibility

Every number is reproducible with one command on data anyone can download. See
[Reproduce](reproduce.md).
