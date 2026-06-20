"""Dataset adapter contract.

A dataset adapter turns a downloaded corpus into a list of cases the runner can
score: an input image for the model to segment, a ground-truth label map in the
dataset's own :class:`~segbench.schema.LabelSchema`, and per-case metadata used
for failure-mode slicing (modality, and any demographic / scanner field the
dataset ships).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from segbench.schema import LabelSchema


@dataclass(frozen=True)
class CaseRecord:
    """One scorable case: an image to segment and its ground truth."""

    case_id: str
    image: Path
    gt: Path
    metadata: dict[str, object] = field(default_factory=dict)


class Dataset:
    """Base class for dataset adapters.

    Subclasses set ``name``, ``schema``, ``license`` and ``citation`` and
    implement :meth:`ensure_available` (idempotent download) and :meth:`cases`.
    """

    name: str
    schema: LabelSchema
    license: str
    citation: str
    #: True only if the ground-truth labels can be redistributed; the leaderboard
    #: never re-hosts images, it publishes derived scores.
    redistribute_scores_ok: bool = True

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def ensure_available(self, n: int) -> None:  # pragma: no cover - I/O
        raise NotImplementedError

    def cases(self, n: int | None = None) -> list[CaseRecord]:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "schema": self.schema.name,
            "license": self.license,
            "citation": self.citation,
            "organs": [{"id": o.id, "name": o.name} for o in self.schema.organs],
        }
