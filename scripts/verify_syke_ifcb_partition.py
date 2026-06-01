"""
Standalone sanity check for the Dirichlet partition on syke-ifcb.

Replicates the exact partition logic from fedDINO_all_eval.py without loading
any model or touching the GPU. Reports per-cell statistics that matter for
federated SimCLR:

  - Client size distribution (min / median / max images per client)
  - Per-client class diversity (how many distinct classes each client has)
  - Empty / near-empty client count (clients with <2 images can't form
    positive pairs and effectively skip the round)
  - Per-round expected work at the sampled participation rate

Run with:
  python scripts/verify_syke_ifcb_partition.py
"""
import os
import sys
import numpy as np

# Ensure we can import sibling modules even when run from another cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets_extra import HFImageDataset, HF_DATASETS


def partition_dirichlet(targets: np.ndarray, num_clients: int, alpha: float, seed: int):
    """Exact copy of FedSimCLRTrainer._partition_dirichlet (unbalanced)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(targets)
    num_cls = len(np.unique(labels))
    idx_by_cls = [np.where(labels == c)[0] for c in range(num_cls)]
    client_idx = [[] for _ in range(num_clients)]
    for c in range(num_cls):
        rng.shuffle(idx_by_cls[c])
        props = rng.dirichlet([alpha] * num_clients)
        props = (np.cumsum(props) * len(idx_by_cls[c])).astype(int)[:-1]
        splits = np.split(idx_by_cls[c], props)
        for cid, s in enumerate(splits):
            client_idx[cid].extend(s.tolist())
    for cid in range(num_clients):
        rng.shuffle(client_idx[cid])
    return client_idx


def describe(client_idx, targets, num_clients, client_frac, label):
    sizes = np.array([len(c) for c in client_idx])
    classes_per = np.array([len(set(int(targets[i]) for i in c)) for c in client_idx])

    print(f"\n--- {label}: num_clients={num_clients}, alpha=0.3, client_frac={client_frac} ---")
    print(f"  total images: {sizes.sum()}")
    print(f"  per-client images:   min={sizes.min()}  median={int(np.median(sizes))}  "
          f"mean={sizes.mean():.1f}  max={sizes.max()}  std={sizes.std():.1f}")
    print(f"  per-client classes:  min={classes_per.min()}  median={int(np.median(classes_per))}  "
          f"mean={classes_per.mean():.2f}  max={classes_per.max()}")
    n_empty = int((sizes == 0).sum())
    n_singleton = int((sizes == 1).sum())
    n_lt_2 = n_empty + n_singleton
    print(f"  clients with 0 images:        {n_empty}  ({100*n_empty/num_clients:.1f}%)")
    print(f"  clients with 1 image:         {n_singleton}  ({100*n_singleton/num_clients:.1f}%)")
    print(f"  clients with <2 images:       {n_lt_2}  (no positive pair possible)")
    sampled = max(1, int(client_frac * num_clients))
    expected_active = sampled * (sizes >= 2).sum() / num_clients
    print(f"  sampled clients per round:    {sampled}  "
          f"(expected {expected_active:.1f} usable with >=2 images)")
    # Class coverage in a typical round: simulate one sample of `sampled` clients.
    rng = np.random.default_rng(0)
    chosen = rng.choice(num_clients, size=sampled, replace=False)
    classes_seen = set()
    for cid in chosen:
        classes_seen.update(int(targets[i]) for i in client_idx[cid])
    print(f"  classes seen by sampled round (one example draw): "
          f"{len(classes_seen)} / {len(set(targets.tolist()))}")


def main():
    name = "syke_ifcb"
    print(f"Loading HF dataset registered as '{name}' "
          f"({HF_DATASETS[name].repo_id}, split='train') ...")
    ds = HFImageDataset.from_registry(name, split="train", transform=None)
    targets = np.array(ds.targets)
    n_classes = len(set(targets.tolist()))
    print(f"  total train images: {len(targets)}")
    print(f"  classes:            {n_classes}")
    # Class-frequency summary.
    counts = np.bincount(targets)
    print(f"  class size distribution: min={counts.min()} median={int(np.median(counts))} "
          f"mean={counts.mean():.1f} max={counts.max()}")
    print(f"  most imbalanced ratio: {counts.max() / max(1, counts.min()):.1f}x")

    cells = [
        (200,  0.01, "200C / 1%"),
        (200,  0.05, "200C / 5%"),
        (2000, 0.01, "2000C / 1%"),
        (2000, 0.05, "2000C / 5%"),
    ]
    seeds = [1, 17, 27]

    print("\n" + "=" * 72)
    print("All three seeds (1, 17, 27) summary table.")
    print("Checks: 'lt_2' = clients with <2 images (no positive pair in SimCLR).")
    print("=" * 72)
    header = f"{'cell':<14}{'seed':>6}{'min':>6}{'median':>8}{'max':>6}{'std':>8}" \
             f"{'cls_med':>9}{'empty':>8}{'lt_2':>7}"
    print(header)
    print("-" * len(header))
    any_bad = False
    for num_clients, frac, label in cells:
        for seed in seeds:
            client_idx = partition_dirichlet(targets, num_clients, alpha=0.3, seed=seed)
            sizes = np.array([len(c) for c in client_idx])
            classes_per = np.array([
                len(set(int(targets[i]) for i in c)) for c in client_idx
            ])
            n_empty = int((sizes == 0).sum())
            n_lt_2 = int((sizes < 2).sum())
            flag = "  <- !" if (n_empty > 0 or n_lt_2 > 0) else ""
            if n_lt_2 > 0:
                any_bad = True
            print(f"{label:<14}{seed:>6}{sizes.min():>6}{int(np.median(sizes)):>8}"
                  f"{sizes.max():>6}{sizes.std():>8.1f}{int(np.median(classes_per)):>9}"
                  f"{n_empty:>8}{n_lt_2:>7}{flag}")

    print()
    if any_bad:
        print("WARNING: at least one (cell, seed) combination produced a client with <2 "
              "images. That client cannot form a SimCLR positive pair and will be a "
              "wasted slot if sampled in a round. Consider min-client-size filtering "
              "or a different alpha.")
    else:
        print("OK: across all 4 cells x 3 seeds, every client has >=2 images "
              "(every client can form a SimCLR positive pair if sampled).")

    # Detailed dump for seed=1 only, for inspection.
    print()
    print("=" * 72)
    print("Detailed per-cell stats for seed=1 (representative).")
    print("=" * 72)
    for num_clients, frac, label in cells:
        client_idx = partition_dirichlet(targets, num_clients, alpha=0.3, seed=1)
        describe(client_idx, targets, num_clients, frac, label)


if __name__ == "__main__":
    main()
