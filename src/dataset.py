"""Dataset adapter for the Multi-Document OCR Dataset.

Layout (per doc type): labels/<stem>.json, image/<stem>_page_1.png,
texts/<stem>.txt, pdfs/<stem>.pdf.

Primary key is the *filename stem* (the label file name minus .json), NOT the
`sample_id` field inside the JSON — those can disagree (e.g. invoice files are
named `INV-2026-0001` but may carry a different `sample_id`).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

# Existing dataset lives at the project root (kept read-only). The credit-card
# type is generated into the same dataset folder by src/generate/credit_card.py.
DATASET_ROOT = Path(__file__).resolve().parent.parent / "Multi-Document OCR Dataset"

DOC_TYPES = [
    "bank_statement",
    "receipt",
    "invoice",
    "credit_card_statement",  # generated; absent until src/generate/credit_card.py runs
]

# Types that carry a scorable per-transaction ledger (row-level benchmark).
LEDGER_TYPES = {"bank_statement", "credit_card_statement"}
# Types whose label has a line-item array.
LINE_ITEM_TYPES = {"invoice", "receipt"}


@dataclass(frozen=True)
class DocPaths:
    stem: str          # e.g. "bank_statement_0001" or "INV-2026-0001"
    doc_type: str
    label_path: Path
    image_path: Path   # first/primary page image
    pdf_path: Path
    text_path: Path
    image_paths: tuple = ()   # all page images (multi-page uploads); empty -> single image_path

    def exists(self) -> dict[str, bool]:
        return {
            "label": self.label_path.exists(),
            "image": self.image_path.exists(),
            "pdf": self.pdf_path.exists(),
            "text": self.text_path.exists(),
        }

    def load_label(self) -> dict:
        return json.loads(self.label_path.read_text())


def _type_dir(doc_type: str, root: Path | None = None) -> Path:
    return (root or DATASET_ROOT) / doc_type


def _paths_for_stem(doc_type: str, stem: str, root: Path | None = None) -> DocPaths:
    d = _type_dir(doc_type, root)
    return DocPaths(
        stem=stem,
        doc_type=doc_type,
        label_path=d / "labels" / f"{stem}.json",
        image_path=d / "image" / f"{stem}_page_1.png",
        pdf_path=d / "pdfs" / f"{stem}.pdf",
        text_path=d / "texts" / f"{stem}.txt",
    )


def list_docs(doc_type: str, root: Path | None = None) -> list[DocPaths]:
    """All documents of a type, keyed by label-file stem. Empty if type absent."""
    labels_dir = _type_dir(doc_type, root) / "labels"
    if not labels_dir.is_dir():
        return []
    stems = sorted(p.stem for p in labels_dir.glob("*.json"))
    return [_paths_for_stem(doc_type, s, root) for s in stems]


def get_doc(doc_type: str, stem: str, root: Path | None = None) -> DocPaths:
    return _paths_for_stem(doc_type, stem, root)


def available_types(root: Path | None = None) -> list[str]:
    return [t for t in DOC_TYPES if (_type_dir(t, root) / "labels").is_dir()]


def stratified_sample(
    n_per_type: int,
    seed: int = 42,
    types: list[str] | None = None,
    root: Path | None = None,
) -> list[DocPaths]:
    """Fixed pseudo-random subset with up to `n_per_type` docs per type.

    Deterministic given (seed, n_per_type) so benchmark runs are reproducible.
    """
    rng = random.Random(seed)
    types = types or available_types(root)
    out: list[DocPaths] = []
    for t in types:
        docs = list_docs(t, root)
        if len(docs) <= n_per_type:
            out.extend(docs)
        else:
            out.extend(rng.sample(docs, n_per_type))
    return out


def dataset_summary(root: Path | None = None) -> dict[str, dict]:
    """Per-type counts and pairing completeness — for quick sanity checks."""
    summary: dict[str, dict] = {}
    for t in DOC_TYPES:
        docs = list_docs(t, root)
        if not docs:
            continue
        missing = {"label": 0, "image": 0, "pdf": 0, "text": 0}
        for d in docs:
            for k, ok in d.exists().items():
                if not ok:
                    missing[k] += 1
        summary[t] = {"count": len(docs), "missing": missing}
    return summary


if __name__ == "__main__":
    import pprint

    print("DATASET_ROOT:", DATASET_ROOT, "exists:", DATASET_ROOT.exists())
    print("available types:", available_types())
    pprint.pprint(dataset_summary())
    # spot-check joins for the two tricky naming schemes
    for t in ("bank_statement", "invoice"):
        docs = list_docs(t)
        if docs:
            d = docs[0]
            print(f"\n[{t}] first stem={d.stem}  exists={d.exists()}")
    samp = stratified_sample(3)
    print("\nstratified_sample(3):", len(samp), "docs;",
          [(d.doc_type, d.stem) for d in samp])
