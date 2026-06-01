#!/usr/bin/env python3
"""
Aggregator for Table 4: federated SimCLR trained on WCO L4 IFCB, evaluated on
7 plankton datasets (wco_l4_ifcb_pml in-domain + 6 cross-domain plankton sets).

Reads fl_results/fed_unsup_simclr_<backbone_tag>_wco_l4_ifcb_pml_<run_name>_<ts>.json,
groups by (backbone_tag, cell, eval_dataset), reports mean +/- std across seeds.

Mirrors aggregate_table3_compare.py one-for-one; only the train-dataset name
and eval-dataset list differ.
"""
import glob, json, re, os
from collections import defaultdict
import numpy as np

RESULTS_DIR = "fl_results"
TRAIN_DATASET = "wco_l4_ifcb_pml"

# Set DROP_SYKE_IFCB=1 to omit syke_ifcb from all Table 4 displays
# (per-backbone tables, paper-format table, side-by-side, full breakdown).
DROP_SYKE_IFCB = os.environ.get("DROP_SYKE_IFCB", "0") == "1"

BACKBONES = ["resnet18", "clip_b16", "dino_s16", "dino_b16", "tips_b14"]
BACKBONE_LABELS = {
    "resnet18":  "ResNet-18",
    "clip_b16":  "CLIP-B/16",
    "dino_s16":  "DINOv3-S/16",
    "dino_b16":  "DINOv3-B/16",
    "tips_b14":  "TIPS-B/14",
}

# Filename pattern: fed_unsup_simclr_<tag>_<TRAIN_DATASET>_(optional <tag>_)<cell>_seed<S>_<ts>.json
_TAG_ALTS = "|".join(BACKBONES)
BACKBONE_RE = re.compile(
    rf"fed_unsup_simclr_({_TAG_ALTS})_{re.escape(TRAIN_DATASET)}_(?:(?:{_TAG_ALTS})_)?(\d+C_\d+pct)_seed(\d+)_"
)

CELLS = ["200C_1pct", "200C_5pct", "2000C_1pct", "2000C_5pct"]
# Order controls table column order. In-domain first, then 6 cross-domain plankton.
EVAL_DATASETS = [
    # in-domain
    "wco_l4_ifcb_pml",
    # cross-domain plankton (OOD)
    "syke_ifcb",
    "syke_zooscan",
    "plankto_share",
    "daplankton_lab_ifcb",
    "daplankton_lab_cs",
    "daplankton_lab_fc",
]
if DROP_SYKE_IFCB:
    EVAL_DATASETS = [ev for ev in EVAL_DATASETS if ev != "syke_ifcb"]

EVAL_DATASET_LABELS = {
    "wco_l4_ifcb_pml":      "WCO-L4-IFCB",
    "syke_ifcb":            "SYKE-IFCB",
    "syke_zooscan":         "SYKE-ZooScan",
    "plankto_share":        "PlanktoShare",
    "daplankton_lab_ifcb":  "DAP-IFCB",
    "daplankton_lab_cs":    "DAP-CS",
    "daplankton_lab_fc":    "DAP-FC",
}

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
        ev = entry.get("eval_dataset", TRAIN_DATASET)
        for k, _ in EVAL_METRIC_KEYS:
            if k in entry and entry[k] is not None:
                by_eval[ev][k] = float(entry[k])
    for entry in d.get("unsup_results", []):
        ev = entry.get("eval_dataset", TRAIN_DATASET)
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
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    seed_seen = defaultdict(lambda: defaultdict(set))

    for path in files:
        m = BACKBONE_RE.search(os.path.basename(path))
        if not m:
            continue
        backbone, cell, seed = m.group(1), m.group(2), int(m.group(3))
        key = (cell, seed)
        if key in seed_seen[backbone][cell]:
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
    print("Table 4: federated SimCLR trained on WCO L4 IFCB, evaluated across 7 plankton datasets.")
    print("Backbones: " + ", ".join(BACKBONE_LABELS[bb] for bb in BACKBONES))
    print(f"Training: {TRAIN_DATASET} with alpha=0.3 unbalanced Dirichlet partition.")
    print("Seeds: {1, 17, 27}.")
    print("=" * 120)

    for bb in BACKBONES:
        print(f"\n[{BACKBONE_LABELS[bb]}] Completed runs per cell:")
        for cell in CELLS:
            seeds = sorted({s for (_, s) in seed_seen[bb][cell]})
            print(f"  {cell}: seeds={seeds}")

    for bb in BACKBONES:
        if not grouped[bb]:
            continue
        print(f"\n[{BACKBONE_LABELS[bb]}] Linear accuracy (%):")
        header = f"{'Cell':<14}" + "".join(f"{EVAL_DATASET_LABELS[ev]:>20}" for ev in EVAL_DATASETS)
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
    # Split into in-domain vs cross-domain plankton, to fit terminal width.
    # ============================================================
    PARTICIPATIONS = [("1%", "1pct"), ("5%", "5pct")]
    SCALES = ["200C", "2000C"]
    IN_DOMAIN = ["wco_l4_ifcb_pml"]
    OOD = ["syke_ifcb", "syke_zooscan", "plankto_share", "daplankton_lab_ifcb", "daplankton_lab_cs", "daplankton_lab_fc"]
    if DROP_SYKE_IFCB:
        OOD = [ev for ev in OOD if ev != "syke_ifcb"]
    pct_label_w = 12
    method_w = 17

    def print_paper_table(title: str, ev_names: list[str], col_headers: list[str]):
        cell_w = 15
        line_w = pct_label_w + method_w + cell_w * len(ev_names) * len(SCALES)
        print()
        print("=" * line_w)
        print(title)
        print("=" * line_w)
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

    # Combined: in-domain first column, then cross-domain plankton.
    print_paper_table(
        "PAPER TABLE 4 FORMAT  --  Linear accuracy (%); leftmost column = in-domain (WCO-L4-IFCB)",
        IN_DOMAIN + OOD,
        [EVAL_DATASET_LABELS[ev] for ev in IN_DOMAIN + OOD],
    )

    # ============================================================
    # Side-by-side: all backbones, per eval dataset (compact view)
    # ============================================================
    print()
    print("Side-by-side -- Linear accuracy (%):")
    print("=" * 120)
    for ev in EVAL_DATASETS:
        print(f"\nEval on {EVAL_DATASET_LABELS[ev]}:")
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
            print(f"\n--- {bb.upper()} | eval on {EVAL_DATASET_LABELS[ev]} ---")
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
