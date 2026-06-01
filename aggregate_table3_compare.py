#!/usr/bin/env python3
"""
Side-by-side aggregator: compare FedCLIP-B/16, FedDINOv3-S/16, FedDINOv3-B/16,
and FedTIPS-B/14 (Table 3 row + cross-domain OOD eval columns).

Reads fl_results/fed_unsup_simclr_<backbone_tag>_cifar10_<run_name>_<ts>.json,
groups by (backbone_tag, cell, eval_dataset), reports mean +/- std across seeds.

Recognized backbone tags: clip_b16, dino_s16, dino_b16, tips_b14.

Protocol notes (kept uniform across backbones, matches the paper's text):
  - Input size: 224x224 for all backbones.
      CLIP-B/16 native is 224 (exact match).
      DINOv3 ViT-S/16 and ViT-B/16 native is 256 (positional embeddings interpolated; benign).
      TIPSv2 B/14 native is 448 (running at half-resolution; deliberate to keep
      compute uniform with the other two and to match the paper's image_size=224).
  - Normalization: ImageNet stats (0.485, 0.456, 0.406)/(0.229, 0.224, 0.225)
    for all backbones. Matches the paper's Section 4.2 ("ImageNet normalisation").
  - Readout: mean over all output tokens (CLS + register + patch) at the last
    transformer layer (post-LN). DINO ViT-S has embed_dim=384; the projector adapts.
"""
import glob, json, re, os
from collections import defaultdict
import numpy as np

RESULTS_DIR = "fl_results"

# Backbones recognized by the aggregator. Order controls table column order.
BACKBONES = ["resnet18", "clip_b16", "dino_s16", "dino_b16", "tips_b14"]
BACKBONE_LABELS = {
    "resnet18":  "ResNet-18",
    "clip_b16":  "CLIP-B/16",
    "dino_s16":  "DINOv3-S/16",
    "dino_b16":  "DINOv3-B/16",
    "tips_b14":  "TIPS-B/14",
}

# Filename pattern: fed_unsup_simclr_<tag>_cifar10_(optional <tag>_)<cell>_seed<S>_<ts>.json
# The optional second occurrence comes from the driver's --run_name including the tag.
_TAG_ALTS = "|".join(BACKBONES)
BACKBONE_RE = re.compile(
    rf"fed_unsup_simclr_({_TAG_ALTS})_cifar10_(?:(?:{_TAG_ALTS})_)?(\d+C_\d+pct)_seed(\d+)_"
)

CELLS = ["200C_1pct", "200C_5pct", "2000C_1pct", "2000C_5pct"]
# In-domain + out-of-domain eval datasets. Order controls column order in tables.
EVAL_DATASETS = [
    # in-domain
    "cifar10", "cifar100", "tiny",
    # out-of-domain (HF-loaded)
    "whoi_plankton", "eurosat", "bone_marrow", "wikiart_style",
]

EVAL_METRIC_KEYS = [("linear_accuracy", "Linear"), ("eval_only_accuracy", "EvalOnly")]
UNSUP_METRIC_KEYS = [
    ("knn_accuracy", "kNN"),
    ("cluster_accuracy", "Cluster"),
    ("prototype_accuracy", "Prototype"),
    ("silhouette_score", "Silhouette"),
]

def load_final_by_eval(path):
    with open(path) as f:
        d = json.load(f)
    by_eval = defaultdict(dict)
    for entry in d.get("eval_results", []):
        ev = entry.get("eval_dataset", "cifar10")
        for k, _ in EVAL_METRIC_KEYS:
            if k in entry and entry[k] is not None:
                by_eval[ev][k] = float(entry[k])
    for entry in d.get("unsup_results", []):
        ev = entry.get("eval_dataset", "cifar10")
        for k, _ in UNSUP_METRIC_KEYS:
            if k in entry and entry[k] is not None:
                by_eval[ev][k] = float(entry[k])
    return by_eval

def fmt(values, pct=True):
    if not values:
        return "    -    "
    arr = np.asarray(values, dtype=float)
    mean = arr.mean() * (100 if pct else 1)
    std = arr.std(ddof=0) * (100 if pct else 1)
    return f"{mean:6.2f} +/- {std:4.2f}" if pct else f"{mean:6.3f} +/- {std:5.3f}"

def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "fed_unsup_simclr_*.json")))
    # grouped[backbone][cell][eval_ds][metric] = list across seeds
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    seed_seen = defaultdict(lambda: defaultdict(set))

    for path in files:
        m = BACKBONE_RE.search(os.path.basename(path))
        if not m:
            continue
        backbone, cell, seed = m.group(1), m.group(2), int(m.group(3))
        key = (cell, seed)
        if key in seed_seen[backbone][cell]:
            # de-dup latest run for the same (backbone,cell,seed)
            for ev in grouped[backbone][cell]:
                for k in list(grouped[backbone][cell][ev].keys()):
                    if grouped[backbone][cell][ev][k]:
                        grouped[backbone][cell][ev][k] = grouped[backbone][cell][ev][k][:-1]
        seed_seen[backbone][cell].add(key)
        per_eval = load_final_by_eval(path)
        for ev, mdict in per_eval.items():
            for k, v in mdict.items():
                grouped[backbone][cell][ev][k].append(v)

    print()
    print("Table 3 reproduction + cross-domain OOD eval.")
    print("Backbones: " + ", ".join(BACKBONE_LABELS[bb] for bb in BACKBONES))
    print("Training: CIFAR-10 with alpha=0.3 unbalanced Dirichlet partition.")
    print("Seeds: {1, 17, 27}.")
    print("=" * 120)

    # Per-backbone seed inventory
    for bb in BACKBONES:
        print(f"\n[{BACKBONE_LABELS[bb]}] Completed runs per cell:")
        for cell in CELLS:
            seeds = sorted({s for (_, s) in seed_seen[bb][cell]})
            print(f"  {cell}: seeds={seeds}")

    # Paper-style row: Linear accuracy per backbone
    for bb in BACKBONES:
        if not grouped[bb]:
            continue
        print(f"\n[{BACKBONE_LABELS[bb]}] Linear accuracy (%):")
        header = f"{'Cell':<14}" + "".join(f"{ev:>20}" for ev in EVAL_DATASETS)
        print(header)
        print("-" * len(header))
        for cell in CELLS:
            row = f"{cell:<14}"
            for ev in EVAL_DATASETS:
                vals = grouped[bb][cell].get(ev, {}).get("linear_accuracy", [])
                row += f"{fmt(vals):>20}"
            print(row)

    # ============================================================
    # Paper-style tables (Participation% x Method, columns = datasets x scales)
    # Split into two tables to fit reasonable terminal width:
    #   In-domain:  CIFAR10, CIFAR100, Tiny-IN  (matches paper Table 3)
    #   OOD:        whoi_plankton, eurosat, bone_marrow, wikiart_style
    # ============================================================
    PARTICIPATIONS = [("1%", "1pct"), ("5%", "5pct")]
    SCALES = ["200C", "2000C"]
    IN_DOMAIN = ["cifar10", "cifar100", "tiny"]
    OOD = ["whoi_plankton", "eurosat", "bone_marrow", "wikiart_style"]
    pct_label_w = 12
    method_w = 17

    def print_paper_table(title: str, ev_names: list[str], col_headers: list[str]):
        cell_w = 15
        line_w = pct_label_w + method_w + cell_w * len(ev_names) * len(SCALES)
        print()
        print("=" * line_w)
        print(title)
        print("=" * line_w)
        # Two-row header.
        header1 = f"{'':<{pct_label_w}}{'':<{method_w}}"
        for ev_label in col_headers:
            header1 += f"{ev_label:^{cell_w * 2}}"
        print(header1)
        header2 = f"{'Particip.':<{pct_label_w}}{'Method':<{method_w}}"
        for _ in range(len(ev_names)):
            header2 += f"{'200C':>{cell_w}}{'2000C':>{cell_w}}"
        print(header2)
        print("-" * line_w)
        for part_label, part_key in PARTICIPATIONS:
            print(f"{part_label:<{pct_label_w}}", end="")
            first = True
            for bb in BACKBONES:
                if not first:
                    print(f"{'':<{pct_label_w}}", end="")
                first = False
                method_label = f"Fed{BACKBONE_LABELS[bb]}"
                print(f"{method_label:<{method_w}}", end="")
                for ev in ev_names:
                    for scale in SCALES:
                        cell = f"{scale}_{part_key}"
                        vals = grouped[bb][cell].get(ev, {}).get("linear_accuracy", [])
                        print(f"{fmt(vals):>{cell_w}}", end="")
                print()
            print("-" * line_w)

    print_paper_table(
        "PAPER TABLE 3 FORMAT  --  Linear accuracy (%) on in-domain evaluation",
        IN_DOMAIN,
        ["CIFAR10", "CIFAR100", "Tiny-IN"],
    )
    print_paper_table(
        "OOD EVAL  --  Linear accuracy (%) on out-of-distribution datasets",
        OOD,
        ["WHOI", "EuroSAT", "Bone-Marrow", "WikiArt"],
    )

    # ============================================================
    # Side-by-side: all backbones, per eval dataset (compact view)
    # ============================================================
    print()
    print("Side-by-side -- Linear accuracy (%):")
    print("=" * 120)
    for ev in EVAL_DATASETS:
        print(f"\nEval on {ev}:")
        col_w = 18
        header = f"{'Cell':<14}" + "".join(f"{BACKBONE_LABELS[bb]:>{col_w}}" for bb in BACKBONES) + f"{'best':>14}"
        print(header)
        print("-" * len(header))
        for cell in CELLS:
            row = f"{cell:<14}"
            means = {}
            for bb in BACKBONES:
                vals = grouped[bb][cell].get(ev, {}).get("linear_accuracy", [])
                row += f"{fmt(vals):>{col_w}}"
                if vals:
                    means[bb] = float(np.mean(vals))
            best = max(means, key=means.get) if means else ""
            row += f"{BACKBONE_LABELS.get(best, ''):>14}"
            print(row)

    # Full breakdown for each backbone
    all_metric_keys = [k for k, _ in EVAL_METRIC_KEYS] + [k for k, _ in UNSUP_METRIC_KEYS]
    all_metric_names = [n for _, n in EVAL_METRIC_KEYS] + [n for _, n in UNSUP_METRIC_KEYS]
    for bb in BACKBONES:
        if not grouped[bb]:
            continue
        for ev in EVAL_DATASETS:
            print(f"\n--- {bb.upper()} | eval on {ev} ---")
            header = f"{'Cell':<14}" + "".join(f"{name:>16}" for name in all_metric_names) + f"{'n':>5}"
            print(header)
            print("-" * len(header))
            for cell in CELLS:
                row = f"{cell:<14}"
                for k in all_metric_keys:
                    vals = grouped[bb][cell].get(ev, {}).get(k, [])
                    pct = (k != "silhouette_score")
                    row += f"{fmt(vals, pct=pct):>16}"
                row += f"{len(seed_seen[bb][cell]):>5}"
                print(row)

if __name__ == "__main__":
    main()
