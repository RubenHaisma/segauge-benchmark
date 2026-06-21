# The segauge medical segmentation leaderboard

**Every score with a confidence interval and a ranking-stability test. Sliced by
failure mode. On public data. Reproducible with one command.**

Most segmentation leaderboards report a single Dice number with no interval, so
you cannot tell a real lead from sampling noise. The medical-imaging metrics
community has shown this repeatedly: removing one test case flips most teams'
ranks in a majority of challenges, and the reported "winner" often sits inside
the runner-up's confidence interval. This leaderboard is that critique turned
into a running tool.

!!! warning "Preview (GPU via Modal) - whole-kidney on KiTS23, N=20"
    This is an early, small-N preview that demonstrates the methodology end to end on real, public data. The wide intervals are the point: they show how little a bare leaderboard number tells you at this sample size. Scaling to more models and organs is the next step.

**Dataset:** kits23 (20 cases, ground-truth labels CC-BY-NC-SA-4.0). Scored with [segauge](https://github.com/RubenHaisma/segauge) v0.2.0 (95% bootstrap CI, 2000 resamples, seed 0).
See [Methodology](methodology.md) and [Reproduce](reproduce.md).

A leaderboard you can run yourself: `pip install segauge` and one `segbench run`.

## Kidney

| Model | Fair | Dice | HD95 (mm) | NSD | ASSD (mm) |
|---|---|---|---|---|---|
| TotalSegmentator | yes | 0.904 [0.881, 0.922] | 6.13 [4.37, 8.24] | 0.824 [0.778, 0.863] | 1.5 [1.14, 1.95] |
| CT-FM | yes | 0.891 [0.848, 0.92] | 7.24 [4.88, 10.4] | 0.805 [0.752, 0.85] | 1.79 [1.25, 2.47] |
| MOOSE | yes | 0.905 [0.879, 0.928] | 6.58 [4.24, 9.36] | 0.834 [0.786, 0.874] | 1.54 [1.07, 2.1] |

**Ranking stability** (by Dice, paired bootstrap over n=20 cases):

- MOOSE: 0.905 (P(rank 1) = 64%, mean rank 1.4 [1, 3])
- TotalSegmentator: 0.904 (P(rank 1) = 36%, mean rank 1.7 [1, 3])
- CT-FM: 0.891 (P(rank 1) = 0%, mean rank 2.9 [2, 3])

**Statistical separability:**

- **MOOSE and TotalSegmentator are not statistically separable** (Δ=0.00137, CI includes 0)
- **MOOSE and CT-FM are not statistically separable** (Δ=0.0148, CI includes 0)
- **TotalSegmentator and CT-FM are not statistically separable** (Δ=0.0135, CI includes 0)

*Pairwise intervals use a Bonferroni family-wise correction over 3 comparisons (each computed at 98.3% so the family holds 95%).*

---

*Contamination policy:* a model is only ranked on a dataset it was not trained on. Cells marked "no" are shown for context but excluded from ranking. *Dataset citation:* Heller et al., The KiTS23 Challenge (2023). github.com/neheller/kits23 *Source:* https://github.com/neheller/kits23.
