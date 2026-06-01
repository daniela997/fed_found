#!/usr/bin/env python3
"""
Upload an ImageFolder-format dataset to the HuggingFace Hub with a
project-oceania-style dataset card.

Inspired by planktonzilla's DatasetImporter but lightweight and standalone.
Produces a Convention A schema: `{image: Image, label: ClassLabel[N]}`,
single train split. Rich taxonomic enrichment (Convention B) is out of scope
for this script.

Usage:
  python upload_to_hf.py hf_upload_configs/<name>.yaml [--dry-run]

Required: HF_TOKEN in env, or set `hf_token` in the YAML.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from datasets import load_dataset
from huggingface_hub import DatasetCard, DatasetCardData


# ============================================================
# Config schema
# ============================================================

@dataclass
class UploadConfig:
    # --- required ---
    data_dir: str                          # path to ImageFolder root
    hf_dataset_name: str                   # name on HF (without org prefix)
    hf_org_name: str                       # HF user/org

    # --- recommended metadata for the dataset card ---
    pretty_name: str = ""
    description: str = ""
    license: str = ""                      # e.g. "cc-by-nc-4.0", "mit"
    source_url: str = ""                   # original dataset homepage
    citation_apa: str = ""
    citation_bibtex: str = ""
    paperswithcode_id: str = ""
    arxiv_id: str = ""

    # --- behavior flags ---
    hf_private: bool = True
    hf_token: Optional[str] = None         # falls back to $HF_TOKEN
    num_proc: int = 8
    push_to_hub_retries: int = 5
    check_image_file_integrity: bool = False

    # --- splits ---
    # Three mutually-exclusive options:
    #   (a) No split: single 'train' split with all images. Default behavior.
    #   (b) `split_ratios=[...]`: per-class shuffle + sequential split. Two
    #       values -> train/test. Three -> train/validation/test. Ratios must
    #       sum to 1.0. Replicates syke-pic's ModelData logic.
    #   (c) `split_file: <path>`: read explicit per-class image lists from a
    #       JSON file in the schema:
    #          {
    #            "categories": {"<id>": "<class_name>", ...},
    #            "images":     {"<id>": {"<split>": ["fname.jpg", ...], ...}, ...}
    #          }
    #       Split names in the JSON can be train/valid/validation/test; "valid"
    #       is normalized to "validation" on the way in. Filenames are relative
    #       to <data_dir>/<class_name>/<filename>.
    split_ratios: Optional[list[float]] = None
    split_file: Optional[str] = None
    split_random_seed: int = 24
    # Class-size filtering (matches syke-pic min_N/max_N).
    min_N: Optional[int] = None
    max_N: Optional[int] = None
    # Class names to skip (e.g. "Unclassifiable").
    exclude_classes: list[str] = field(default_factory=list)

    # --- extra free-form tags (HF dataset card supports these natively) ---
    task_categories: list[str] = field(default_factory=lambda: ["image-classification"])
    task_ids: list[str] = field(default_factory=lambda: ["multi-class-image-classification"])
    annotations_creators: list[str] = field(default_factory=lambda: ["expert-generated"])
    language: list[str] = field(default_factory=lambda: ["en"])
    tags: list[str] = field(default_factory=list)  # free-form tags

    def __post_init__(self):
        self.data_dir = str(Path(self.data_dir).expanduser().resolve())
        if not Path(self.data_dir).is_dir():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")
        if not self.hf_dataset_name:
            raise ValueError("hf_dataset_name is required")
        if not self.hf_org_name:
            raise ValueError("hf_org_name is required")
        if self.hf_token is None:
            # Resolution order:
            #   1. HF_TOKEN env var
            #   2. HUGGING_FACE_HUB_TOKEN env var (legacy name)
            #   3. huggingface_hub's own token resolution (handles HF_HOME and
            #      the cached token from `hf auth login` / `huggingface-cli login`).
            self.hf_token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            )
            if not self.hf_token:
                try:
                    # huggingface_hub >= 0.19: `get_token()` is the canonical API.
                    # Older versions: fall back to `HfFolder.get_token()`.
                    try:
                        from huggingface_hub import get_token
                        self.hf_token = get_token()
                    except ImportError:
                        from huggingface_hub import HfFolder
                        self.hf_token = HfFolder.get_token()
                except Exception:
                    self.hf_token = None
        if self.split_ratios is not None and self.split_file is not None:
            raise ValueError("Set either split_ratios OR split_file, not both.")
        if self.split_ratios is not None:
            if len(self.split_ratios) not in (2, 3):
                raise ValueError(f"split_ratios must have 2 or 3 values, got {self.split_ratios}")
            s = sum(self.split_ratios)
            if not (0.999 <= s <= 1.001):
                raise ValueError(f"split_ratios must sum to 1.0, got {s}")
        if self.split_file is not None:
            sf = Path(self.split_file).expanduser().resolve()
            if not sf.is_file():
                raise FileNotFoundError(f"split_file does not exist: {sf}")
            self.split_file = str(sf)


# ============================================================
# Dataset card template
# ============================================================

DATACARD_TEMPLATE = """\
---
{{ card_data }}
---
# {{ pretty_name | default("Dataset", true) }}

{{ description | default("(no description provided)", true) }}

- **Source:** {{ source_url | default("(not specified)", true) }}
- **License:** {{ license | default("(not specified)", true) }}

## Dataset statistics

- **Splits:** {{ splits_line }}
- **Number of classes:** {{ n_classes }}
- **Per-channel mean (RGB, train):** {{ mean }}
- **Per-channel std (RGB, train):** {{ std }}

### Label distribution

{{ label_histogram }}

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{{ hf_org_name }}/{{ hf_dataset_name }}")
```

## Citation

{{ citation_apa | default("(citation not provided)", true) }}

### BibTeX

```bibtex
{{ citation_bibtex | default("(bibtex not provided)", true) }}
```
"""


# ============================================================
# Helpers
# ============================================================

def is_valid_image_file(path: Path) -> bool:
    from PIL import Image
    try:
        with Image.open(path) as img:
            img.crop((0, 0, 1, 1))
        return True
    except Exception:
        return False


def compute_label_histogram(ds, label_feature) -> tuple[str, dict[str, int]]:
    """Return (markdown table, {class_name: count})."""
    labels = ds["label"]
    idxs, counts = np.unique(labels, return_counts=True)
    rows = []
    counts_dict = {}
    for i, c in zip(idxs, counts):
        name = label_feature.int2str(int(i))
        counts_dict[name] = int(c)
        rows.append(f"| {int(i):>4} | {name} | {int(c)} |")
    table = "| Class ID | Class name | Count |\n|---|---|---|\n" + "\n".join(rows)
    return table, counts_dict


def compute_mean_std(ds, sample_size: int = 2000) -> tuple[list[float], list[float]]:
    """Compute per-channel mean/std on a subsample. Fast approximation."""
    n = min(sample_size, len(ds))
    idxs = np.random.default_rng(seed=0).choice(len(ds), n, replace=False)
    sums = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for i in idxs:
        img = ds[int(i)]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float64) / 255.0  # [H, W, 3]
        sums += arr.sum(axis=(0, 1))
        sums_sq += (arr ** 2).sum(axis=(0, 1))
        count += arr.shape[0] * arr.shape[1]
    mean = sums / count
    var = sums_sq / count - mean ** 2
    std = np.sqrt(np.clip(var, 0, None))
    return mean.tolist(), std.tolist()


# ============================================================
# Main upload flow
# ============================================================

def _gather_class_paths(
    cfg: UploadConfig,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Walk the data_dir and return:
      - paths_by_class: {class_name: [sorted absolute paths]}
      - classes: alphabetically sorted list of class names
    Honors min_N, max_N, exclude_classes. Image filename extensions are
    detected dynamically (png/jpg/jpeg/webp), matching the syke-pic loader
    which uses `.png` only (we generalize since other datasets may differ).
    """
    import random as _random
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths_by_class: dict[str, list[str]] = {}
    root = Path(cfg.data_dir)
    excluded = set(cfg.exclude_classes or [])

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name in excluded:
            continue
        paths = sorted([
            str(p) for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ])
        if cfg.min_N is not None and len(paths) < cfg.min_N:
            continue
        if cfg.max_N is not None and len(paths) > cfg.max_N:
            _random.seed(cfg.split_random_seed)
            _random.shuffle(paths)
            paths = paths[: cfg.max_N]
            paths.sort()  # restore deterministic order before the split shuffle
        if paths:
            paths_by_class[class_dir.name] = paths

    classes = sorted(paths_by_class.keys())
    return paths_by_class, classes


def _split_paths_per_class(
    paths_by_class: dict[str, list[str]],
    split_ratios: list[float],
    seed: int,
) -> dict[str, list[tuple[str, str]]]:
    """
    Per-class shuffle + sequential split. Matches syke-pic's ModelData logic:
    for each class, seed RNG, shuffle paths, then take train_split fraction,
    then val_split fraction, remainder as test (if 3 ratios).

    Returns: {split_name: [(path, class_name), ...]}
    """
    import random as _random
    if len(split_ratios) == 3:
        names = ["train", "validation", "test"]
        train_split, val_split, _ = split_ratios
    else:
        names = ["train", "test"]
        train_split, _ = split_ratios
        val_split = 0.0

    out: dict[str, list[tuple[str, str]]] = {n: [] for n in names}

    for class_name in sorted(paths_by_class.keys()):
        paths = list(paths_by_class[class_name])
        _random.seed(seed)
        _random.shuffle(paths)
        train_stop = int(round(len(paths) * train_split))
        val_stop = train_stop + int(round(len(paths) * val_split))
        train = paths[:train_stop]
        if len(split_ratios) == 3:
            val = paths[train_stop:val_stop]
            test = paths[val_stop:]
            assert train and val and test, (
                f"'{class_name}' doesn't have enough samples ({len(paths)})."
                " Consider lowering min_N or adjusting split ratios."
            )
            out["train"].extend((p, class_name) for p in train)
            out["validation"].extend((p, class_name) for p in val)
            out["test"].extend((p, class_name) for p in test)
        else:
            test = paths[train_stop:]
            assert train and test, (
                f"'{class_name}' doesn't have enough samples ({len(paths)})."
            )
            out["train"].extend((p, class_name) for p in train)
            out["test"].extend((p, class_name) for p in test)

    # Shuffle each split (matches syke-pic's final shuffle step).
    for name in names:
        _random.seed(seed)
        _random.shuffle(out[name])

    return out


def _splits_from_json(cfg: UploadConfig) -> dict[str, list[tuple[str, str]]]:
    """
    Read explicit per-class split assignments from a JSON file with schema:
        {"categories": {"<id>": "<class_name>", ...},
         "images":     {"<id>": {"<split>": ["fname.jpg", ...], ...}, ...}}
    Returns {split_name: [(absolute_path, class_name), ...]}.

    Split name "valid" is normalized to "validation" to match HF conventions.
    """
    with open(cfg.split_file) as f:
        spec = json.load(f)
    if "categories" not in spec or "images" not in spec:
        raise ValueError(f"split_file {cfg.split_file} missing required keys 'categories' / 'images'")

    excluded = set(cfg.exclude_classes or [])
    out: dict[str, list[tuple[str, str]]] = {}

    for cid, class_name in spec["categories"].items():
        if class_name in excluded:
            continue
        per_class = spec["images"].get(cid, {})
        for raw_split, filenames in per_class.items():
            split_name = "validation" if raw_split == "valid" else raw_split
            class_dir = Path(cfg.data_dir) / class_name
            for fn in filenames:
                p = class_dir / fn
                out.setdefault(split_name, []).append((str(p), class_name))

    return out


def load_imagefolder(cfg: UploadConfig):
    """
    Load dataset(s). Returns a DatasetDict.

    Three paths:
      A) split_ratios=None and split_file=None: single 'train' split via
         HF's imagefolder builder.
      B) split_ratios set:  per-class shuffle + sequential split (matches
         syke-pic ModelData logic), build a DatasetDict with the named splits.
      C) split_file set:    read explicit per-class image-list assignments from
         a JSON file; build DatasetDict with the splits declared in the file.
    """
    from datasets import Dataset, DatasetDict, Image as HFImage, ClassLabel, Features

    if cfg.split_ratios is None and cfg.split_file is None:
        print(f"Loading ImageFolder from {cfg.data_dir} (single 'train' split) ...")
        glob = str(Path(cfg.data_dir) / "*" / "[!._]*")
        ds = load_dataset(
            "imagefolder",
            data_files={"train": glob},
            num_proc=cfg.num_proc,
        )
        return ds

    # Both custom paths emit (split_name -> [(path, class_name), ...]).
    if cfg.split_file is not None:
        print(f"Loading ImageFolder from {cfg.data_dir} with splits from "
              f"{cfg.split_file} ...")
        splits = _splits_from_json(cfg)
        # Class list: include every class that appears in any split.
        classes = sorted({c for items in splits.values() for _, c in items})
    else:
        print(f"Loading ImageFolder from {cfg.data_dir} with custom split ratios "
              f"{cfg.split_ratios} (seed={cfg.split_random_seed}) ...")
        paths_by_class, classes = _gather_class_paths(cfg)
        print(f"  Found {len(classes)} classes; "
              f"{sum(len(v) for v in paths_by_class.values())} total images.")
        splits = _split_paths_per_class(paths_by_class, cfg.split_ratios, cfg.split_random_seed)

    if cfg.exclude_classes:
        print(f"  Excluding classes: {cfg.exclude_classes}")

    class_label = ClassLabel(names=classes)
    features = Features({"image": HFImage(), "label": class_label})

    dd = {}
    for split_name in sorted(splits.keys()):
        items = splits[split_name]
        if not items:
            continue
        images = [{"path": p} for p, _ in items]   # let HF lazy-load
        labels = [class_label.str2int(c) for _, c in items]
        ds = Dataset.from_dict(
            {"image": images, "label": labels},
            features=features,
        )
        dd[split_name] = ds
        print(f"  Split '{split_name}': {len(ds)} images")

    return DatasetDict(dd)


def maybe_validate_images(cfg: UploadConfig):
    """Optionally walk the ImageFolder and remove unreadable files."""
    if not cfg.check_image_file_integrity:
        return
    root = Path(cfg.data_dir)
    n_bad = 0
    n_total = 0
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            n_total += 1
            if not is_valid_image_file(img_path):
                print(f"[invalid] {img_path}")
                img_path.unlink()
                n_bad += 1
    print(f"Validated {n_total} files; removed {n_bad} corrupt.")


def build_card_data(cfg: UploadConfig, stats: dict[str, Any]) -> DatasetCardData:
    """Construct DatasetCardData (YAML frontmatter) from config + stats."""
    return DatasetCardData(
        pretty_name=cfg.pretty_name or cfg.hf_dataset_name,
        license=cfg.license or None,
        task_categories=cfg.task_categories,
        task_ids=cfg.task_ids,
        annotations_creators=cfg.annotations_creators,
        language=cfg.language,
        tags=cfg.tags,
        paperswithcode_id=cfg.paperswithcode_id or None,
        # Custom fields (HF allows arbitrary keys in card_data):
        source_url=cfg.source_url or None,
        citation_apa=cfg.citation_apa or None,
        citation_bibtex=cfg.citation_bibtex or None,
        arxiv_id=cfg.arxiv_id or None,
        dataset_means=stats.get("mean"),
        dataset_stds=stats.get("std"),
    )


def render_card(cfg: UploadConfig, stats: dict[str, Any]) -> DatasetCard:
    """
    Render the dataset card. `DatasetCard.from_template` uses Jinja2 with
    `{{ var }}` substitution. The `{{ card_data }}` placeholder is filled by
    HF with the YAML frontmatter from `DatasetCardData`. All other `{{ var }}`
    placeholders are substituted from kwargs passed below.
    """
    card_data = build_card_data(cfg, stats)
    return DatasetCard.from_template(
        card_data,
        template_str=DATACARD_TEMPLATE,
        pretty_name=cfg.pretty_name or cfg.hf_dataset_name,
        description=cfg.description,
        source_url=cfg.source_url,
        license=cfg.license,
        n_images=stats["n_images"],
        n_classes=stats["n_classes"],
        mean=stats["mean"],
        std=stats["std"],
        label_histogram=stats["label_table"],
        splits_line=stats.get("splits_line", f"train={stats['n_images']}"),
        hf_org_name=cfg.hf_org_name,
        hf_dataset_name=cfg.hf_dataset_name,
        citation_apa=cfg.citation_apa,
        citation_bibtex=cfg.citation_bibtex,
    )


def push_dataset_with_retry(ds, repo_id, token, private, retries):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            ds.push_to_hub(repo_id, token=token, private=private)
            return
        except Exception as e:
            last_err = e
            print(f"[push attempt {attempt}/{retries}] failed: {e!r}")
    raise RuntimeError(f"push_to_hub failed after {retries} retries: {last_err}")


def upload(cfg: UploadConfig, dry_run: bool = False):
    if not cfg.hf_token and not dry_run:
        sys.exit("HF_TOKEN not set. Either export HF_TOKEN or put `hf_token` in YAML.")

    maybe_validate_images(cfg)

    ds = load_imagefolder(cfg)
    # Multi-split aware: compute per-split sizes; histogram + mean/std on the
    # train split (the canonical reference).
    train = ds["train"]
    label_feature = train.features["label"]
    n_classes = len(label_feature.names)
    split_sizes = {name: len(ds[name]) for name in ds}
    n_images_total = sum(split_sizes.values())
    print(f"Loaded splits: {split_sizes} (total {n_images_total}) across {n_classes} classes.")

    print("Computing label histogram (train split) + mean/std (sampled) ...")
    label_table, label_counts = compute_label_histogram(train, label_feature)
    mean, std = compute_mean_std(train)
    mean_fmt = [round(v, 4) for v in mean]
    std_fmt = [round(v, 4) for v in std]

    # Human-friendly splits line for the card.
    splits_line = ", ".join(f"{k}={v}" for k, v in split_sizes.items())

    stats = {
        "n_images": split_sizes.get("train", n_images_total),
        "n_classes": n_classes,
        "label_table": label_table,
        "label_counts": label_counts,
        "mean": mean_fmt,
        "std": std_fmt,
        "splits_line": splits_line,
        "split_sizes": split_sizes,
    }

    repo_id = f"{cfg.hf_org_name}/{cfg.hf_dataset_name}"
    card = render_card(cfg, stats)

    if dry_run:
        print("\n========== DRY RUN ==========")
        print(f"Would push to: {repo_id} (private={cfg.hf_private})")
        print(f"Splits: {splits_line}")
        print(f"Classes: {n_classes}")
        print(f"Mean: {mean_fmt}")
        print(f"Std:  {std_fmt}")
        print("\n----- Card preview -----")
        print(card.content[:2000])
        print("..." if len(card.content) > 2000 else "")
        print("(use without --dry-run to actually push)")
        return

    print(f"\nPushing dataset to {repo_id} ...")
    push_dataset_with_retry(
        ds,
        repo_id=repo_id,
        token=cfg.hf_token,
        private=cfg.hf_private,
        retries=cfg.push_to_hub_retries,
    )

    print(f"Pushing dataset card to {repo_id} ...")
    card.push_to_hub(repo_id, token=cfg.hf_token)

    print(f"\nDone. View at https://huggingface.co/datasets/{repo_id}")


# ============================================================
# CLI
# ============================================================

def load_config(path: str) -> UploadConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a dict")
    return UploadConfig(**data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="Path to YAML config")
    p.add_argument("--dry-run", action="store_true",
                   help="Load, compute stats, render card, but don't push")
    args = p.parse_args()

    cfg = load_config(args.config)
    upload(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
