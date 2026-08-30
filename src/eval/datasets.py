from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

SC_CATEGORY = "composition"


@dataclass
class Sample:
    uid: str
    benchmark: str
    media_type: str            # "image" | "video"
    media_paths: list[str]     # relative to the benchmark root
    question: str
    options: dict[str, str]
    answer: str
    category: str
    meta: dict = field(default_factory=dict)

    def resolve_media(self, root: str | Path) -> list[Path]:
        return [Path(root) / p for p in self.media_paths]


def load_shotbench(root: str | Path) -> list[Sample]:
    tsv = Path(root) / "test.tsv"
    samples = []
    with open(tsv, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            types = json.loads(row["type"])
            samples.append(Sample(
                uid=f"shotbench-{row['index']}",
                benchmark="shotbench",
                media_type=types[0],
                media_paths=list(ast.literal_eval(row["path"])),
                question=row["question"],
                options=json.loads(row["options"]),
                answer=row["answer"].strip(),
                category=row["category"].strip(),
            ))
    return samples


def sc_subset(samples: list[Sample]) -> list[Sample]:
    return [s for s in samples if s.category == SC_CATEGORY]


def by_category(samples: list[Sample]) -> dict[str, list[Sample]]:
    out: dict[str, list[Sample]] = {}
    for s in samples:
        out.setdefault(s.category, []).append(s)
    return out


def load_refineshot(root: str | Path) -> list[Sample]:
    """RefineShot restructures ShotBench options and does not add separate media."""
    raise NotImplementedError(
        "RefineShot is not distributed as a separate benchmark in this package; "
        "use its official option-restructuring code with the ShotBench loader.")


def load_cinetechbench(root: str | Path) -> list[Sample]:
    """CineTechBench composition items are evaluated by scripts/run_ctb.py."""
    raise NotImplementedError(
        "Use scripts/run_ctb.py for the released CineTechBench annotation schema.")
