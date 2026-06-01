"""
Shared dataset adapters used by fedCLIP/fedDINO/fedTIPS _all_eval.py scripts.

- HFImageDataset: wraps a HuggingFace image-classification dataset as a
  torchvision-style Dataset. Plugs into the FL pipeline's Dirichlet partition
  (`.targets`) and the linear-probe eval (returns `(image, label)` tuples).
- HF_DATASETS: registry mapping short user-facing names to HF repo IDs +
  column conventions + which splits to use for train/test.

The registry is the single source of truth for dataset choices; new datasets
are added by editing this file only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import io
import torch
from datasets import load_dataset, Image as HFImage
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from torch.utils.data import Dataset

# ============================================================
# Adapter
# ============================================================

class HFImageDataset(Dataset):
    """
    Torchvision-style Dataset over a HuggingFace image-classification dataset.

    Yields `(PIL.Image, int)` from `__getitem__`. Exposes `.targets` as a
    plain Python list[int] so the FL Dirichlet partition (`np.array(.targets)`)
    works unchanged.

    Use `HFImageDataset.from_repo(short_name, split=...)` to load by registry
    name; or pass `repo_id` and column names directly for ad-hoc use.
    """
    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        transform=None,
        image_col: str = "image",
        label_col: str = "label",
        cache_dir: Optional[str] = None,
    ):
        if cache_dir is None:
            cache_dir = os.environ.get("HF_DATASETS_CACHE")
        self.repo_id = repo_id
        self.split = split
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

        self.ds = load_dataset(repo_id, split=split, cache_dir=cache_dir)

        # Disable HF's automatic image decoding. We'll decode manually in
        # __getitem__ so we can catch corrupt rows without crashing the
        # DataLoader worker. With decode=False, ds[i][image_col] returns a dict
        # {'bytes': ..., 'path': ...} which we decode ourselves.
        if image_col in self.ds.features and isinstance(self.ds.features[image_col], HFImage):
            self.ds = self.ds.cast_column(image_col, HFImage(decode=False))

        # Compute integer targets array.
        labels = self.ds[label_col]
        if len(labels) == 0:
            self.targets: list[int] = []
            self.classes: list = []
        elif isinstance(labels[0], str):
            # String labels: build a stable class index.
            self.classes = sorted(set(labels))
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.targets = [self.class_to_idx[c] for c in labels]
        else:
            # Already integer labels (ClassLabel feature).
            self.targets = list(int(l) for l in labels)
            feat = self.ds.features.get(label_col)
            if feat is not None and hasattr(feat, "names"):
                self.classes = list(feat.names)
            else:
                self.classes = sorted(set(self.targets))

    def __len__(self) -> int:
        return len(self.ds)

    # Class-level counter so we don't spam the log with thousands of identical warnings.
    _n_bad: int = 0

    def __getitem__(self, i: int):
        row = self.ds[int(i)]
        img_field = row[self.image_col]
        img = self._decode_image(img_field, i)
        label = self.targets[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _decode_image(self, img_field, i: int) -> PILImage.Image:
        """
        Decode a (possibly raw-bytes) image record into a PIL.Image in RGB mode.

        If the record is corrupt, return a 1x1 black placeholder. The
        corresponding sample's label is preserved so the linear head still
        sees a class assignment for that index; the embedding will just be
        zero-ish. Logs a warning the first few times.
        """
        try:
            if isinstance(img_field, dict):
                # decode=False mode: dict with 'bytes' and optional 'path'.
                data = img_field.get("bytes")
                if data is None and img_field.get("path"):
                    with open(img_field["path"], "rb") as f:
                        data = f.read()
                img = PILImage.open(io.BytesIO(data))
            else:
                # decode=True path or already-PIL fallback.
                img = img_field
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except (UnidentifiedImageError, OSError, ValueError, AttributeError) as e:
            HFImageDataset._n_bad += 1
            if HFImageDataset._n_bad <= 5:
                print(f"[HFImageDataset] WARN: failed to decode row {i} "
                      f"of {self.repo_id} ({type(e).__name__}: {str(e)[:80]}); "
                      "returning 1x1 black placeholder.")
            return PILImage.new("RGB", (1, 1), color=(0, 0, 0))

    @classmethod
    def from_registry(
        cls,
        name: str,
        split: str = "train",
        transform=None,
        cache_dir: Optional[str] = None,
    ) -> "HFImageDataset":
        if name not in HF_DATASETS:
            raise KeyError(
                f"Unknown HF dataset '{name}'. Known: {sorted(HF_DATASETS)}"
            )
        entry = HF_DATASETS[name]
        return cls(
            repo_id=entry.repo_id,
            split=split,
            transform=transform,
            image_col=entry.image_col,
            label_col=entry.label_col,
            cache_dir=cache_dir,
        )


# ============================================================
# Registry
# ============================================================

@dataclass(frozen=True)
class HFDatasetEntry:
    repo_id: str
    image_col: str = "image"
    label_col: str = "label"
    # Names of train + test splits inside the HF dataset. If a test split is not
    # available, set test_split=None — downstream code can do its own 80/20.
    train_split: str = "train"
    test_split: Optional[str] = None
    # Optional human-readable description (for the registry summary).
    description: str = ""
    # Marker that this dataset is gated; setting it has no effect on loading,
    # but `print_registry` flags it so users know to request access first.
    gated: bool = False


HF_DATASETS: dict[str, HFDatasetEntry] = {
    # ====================
    # Plankton (IFCB)
    # ====================
    "whoi_plankton": HFDatasetEntry(
        repo_id="nf-whoi/whoi-plankton",
        train_split="train",
        test_split="test",
        description="WHOI IFCB benchmark (Orenstein et al. 2015), 100 classes, "
                    "957k images, predefined train/val/test splits.",
    ),
    "syke_ifcb": HFDatasetEntry(
        repo_id="danielaivanova/syke-plankton-ifcb-2022",
        train_split="train",
        test_split="test",
        description="SYKE IFCB 2022 (Kraft et al. 2022). Baltic Sea phytoplankton. "
                    "50 classes, ~63k images. 60/20/20 splits, seed 24.",
    ),
    "wco_l4_ifcb_pml": HFDatasetEntry(
        repo_id="danielaivanova/wco-l4-ifcb-pml",
        train_split="train",
        test_split="test",
        description="WCO L4 IFCB training library (Widdicombe 2026). Western English "
                    "Channel IFCB plankton imagery from Plymouth Marine Laboratory. "
                    "~200 classes, ~90.7k images. 60/20/20 seed 24, min_N=5.",
    ),
    "daplankton_lab_ifcb": HFDatasetEntry(
        repo_id="danielaivanova/daplankton-lab-ifcb",
        train_split="train",
        test_split="test",
        description="DAPlankton_LAB IFCB subset (Batrakhanov et al. 2024). "
                    "15 phytoplankton species, ~16.5k images. 60/20/20 seed 24.",
    ),

    # ====================
    # Other plankton instruments
    # ====================
    "syke_zooscan": HFDatasetEntry(
        repo_id="danielaivanova/syke-plankton-zooscan-2024",
        train_split="train",
        test_split="test",
        description="SYKE ZooScan 2024 (Kareinen et al. 2024). 20 classes (5 test-only "
                    "for open-set recognition), ~22.7k images. Predefined splits.",
    ),
    "plankto_share": HFDatasetEntry(
        repo_id="danielaivanova/plankto-share",
        train_split="train",
        test_split="test",
        description="PlanktoShare (Van Walraven et al., submitted). Pi-10 Plankton "
                    "Imager, North Sea / NE Atlantic. 111 classes, ~52.8k images. "
                    "60/20/20 seed 24.",
    ),
    "daplankton_lab_cs": HFDatasetEntry(
        repo_id="danielaivanova/daplankton-lab-cs",
        train_split="train",
        test_split="test",
        description="DAPlankton_LAB CytoSense subset (Batrakhanov et al. 2024). "
                    "15 phytoplankton species, ~13.2k images. 60/20/20 seed 24.",
    ),
    "daplankton_lab_fc": HFDatasetEntry(
        repo_id="danielaivanova/daplankton-lab-fc",
        train_split="train",
        test_split="test",
        description="DAPlankton_LAB FlowCam subset (Batrakhanov et al. 2024). "
                    "15 phytoplankton species, ~17.8k images. 60/20/20 seed 24.",
    ),

    # ====================
    # Project-oceania (gated; request access via the HF page first)
    # ====================
    "syke_ifcb_2022": HFDatasetEntry(
        repo_id="project-oceania/syke_ifcb_2022",
        train_split="train",
        gated=True,
        description="SYKE IFCB 2022 (Finnish Environment Institute). Baltic Sea. ~63k images, 50 classes.",
    ),
    "flowcamnet": HFDatasetEntry(
        repo_id="project-oceania/flowcamnet",
        train_split="train",
        gated=True,
        description="FlowCAM plankton (project-oceania). ~141k images, 38 classes.",
    ),
    "uvp6net": HFDatasetEntry(
        repo_id="project-oceania/uvp6net",
        train_split="train",
        gated=True,
        description="UVP6 plankton (project-oceania). ~635k images, 54 classes.",
    ),

    # ====================
    # Non-plankton OOD (cross-domain eval columns)
    # ====================
    # Satellite imagery
    "eurosat": HFDatasetEntry(
        repo_id="blanchon/EuroSAT_RGB",
        train_split="train",
        test_split="test",
        description="EuroSAT Sentinel-2 land cover, 10 classes, 27k images. Predefined splits.",
    ),
    # Medical (histology) - bone marrow cells
    "bone_marrow": HFDatasetEntry(
        repo_id="ekim15/bone_marrow_cell_dataset",
        train_split="train",
        test_split="test",
        description="Bone marrow cell microscopy, 21 classes, 170k images. Predefined splits.",
    ),
    # Art (paintings classified by style)
    # WikiArt has 3 label columns; we use `style` (27 classes) as canonical
    # for SSL transfer eval.
    "wikiart_style": HFDatasetEntry(
        repo_id="huggan/wikiart",
        train_split="train",
        test_split=None,           # only train split exists; downstream does 80/20
        label_col="style",
        description="WikiArt paintings by style, 27 classes, ~80k images. Train-only; FL pipeline does 80/20.",
    ),
}


def list_registry() -> str:
    """Return a markdown table of the registered datasets, for diagnostics."""
    rows = ["| Name | Repo | Train | Test | Description |",
            "|---|---|---|---|---|"]
    for name, entry in sorted(HF_DATASETS.items()):
        gated = " (GATED)" if entry.gated else ""
        rows.append(
            f"| {name}{gated} | {entry.repo_id} | {entry.train_split} | "
            f"{entry.test_split or '-'} | {entry.description} |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    # Quick CLI: print the registry table.
    print(list_registry())
