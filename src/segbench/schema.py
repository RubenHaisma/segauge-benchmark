"""Canonical label schemas and the model/dataset interchange contract.

The leaderboard compares models that each emit their *own* label numbering. To
score them against a dataset's ground truth we need one shared vocabulary. That
vocabulary is the **organ name**: a dataset declares which organ each ground-truth
label id means, a model adapter declares which native label id each organ it can
segment, and the runner remaps every model's output into the dataset's id space
by matching on name. A wrong name match silently scores the wrong structure, so
names are normalized and the mapping is published in the methodology page.
"""

from __future__ import annotations

from dataclasses import dataclass

from segauge.core import METRIC_DIRECTION

# Metrics reported on the leaderboard, in display order. Directions come from
# segauge so the two stay in sync.
LEADERBOARD_METRICS = ("dice", "hd95", "nsd", "assd")


def metric_higher_is_better(metric: str) -> bool:
    return METRIC_DIRECTION.get(metric, "up") == "up"


def normalize_organ(name: str) -> str:
    """Canonical form of an organ name for cross-model matching."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class Organ:
    """One labelled structure: its id within a schema and its canonical name."""

    id: int
    name: str

    @property
    def key(self) -> str:
        return normalize_organ(self.name)


@dataclass(frozen=True)
class LabelSchema:
    """An ordered set of organs and their ground-truth label ids in a dataset."""

    name: str
    organs: tuple[Organ, ...]

    def by_name(self) -> dict[str, Organ]:
        return {o.key: o for o in self.organs}

    @property
    def names(self) -> list[str]:
        return [o.name for o in self.organs]


# KiTS23 kidney/tumor segmentation labelling (kidney+tumor+cyst hierarchy).
# Source: github.com/neheller/kits23 (segmentation values 1/2/3).
KITS23_SCHEMA = LabelSchema(
    name="kits23",
    organs=(
        Organ(1, "kidney"),
        Organ(2, "tumor"),
        Organ(3, "cyst"),
    ),
)

# Whole-kidney ("kidney and masses") view: tumor and cyst voxels are anatomically
# inside the kidney, so a whole-organ segmenter like TotalSegmentator must be
# scored against the union of labels 1/2/3, not against parenchyma (label 1)
# alone. This single-organ schema is the fair view for general organ models and
# matches the KiTS "Kidney + Masses" evaluation region.
KITS23_KIDNEY_SCHEMA = LabelSchema(
    name="kits23_kidney",
    organs=(Organ(1, "kidney"),),
)

# AMOS22 abdominal multi-organ labelling (15 organs), used for the Tier-2
# multi-organ run. Ids follow the AMOS dataset.json label map; verify against
# the downloaded dataset before a production run.
AMOS_CT_SCHEMA = LabelSchema(
    name="amos_ct",
    organs=(
        Organ(1, "spleen"),
        Organ(2, "right_kidney"),
        Organ(3, "left_kidney"),
        Organ(4, "gallbladder"),
        Organ(5, "esophagus"),
        Organ(6, "liver"),
        Organ(7, "stomach"),
        Organ(8, "aorta"),
        Organ(9, "inferior_vena_cava"),
        Organ(10, "pancreas"),
        Organ(11, "right_adrenal_gland"),
        Organ(12, "left_adrenal_gland"),
        Organ(13, "duodenum"),
        Organ(14, "bladder"),
        Organ(15, "prostate_uterus"),
    ),
)

SCHEMAS = {s.name: s for s in (KITS23_SCHEMA, AMOS_CT_SCHEMA)}
