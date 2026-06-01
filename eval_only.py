#!/usr/bin/env python3
"""
Evaluation-only driver.

Given a federated SimCLR projector checkpoint (saved at the end of FL training
by fed{CLIP,DINO,TIPS}_all_eval.py), load the backbone + projector and run the
linear-probe / k-NN / clustering / prototype eval suite on a list of eval
datasets. Append results to the matching JSON.

Usage:
  python eval_only.py \\
      --checkpoint fl_results/checkpoints/projector_clip_b16_cifar10_clip_b16_200C_1pct_seed1_<ts>.pt \\
      --json fl_results/fed_unsup_simclr_clip_b16_cifar10_clip_b16_200C_1pct_seed1_<ts>.json \\
      --eval_datasets whoi_plankton,eurosat,bone_marrow,wikiart_style \\
      [--tiny_root /home/daniela/mine/fedDINO/data/tiny_in_view]

The backbone family is inferred from the `backbone_tag` field stored in the
checkpoint. Currently supported tags: clip_b16, dino_s16, dino_b16, tips_b14, resnet18, resnet50.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


# Tag prefix -> backbone module + trainer-class name.
BACKBONE_MODULES = {
    "clip":   "fedCLIP_all_eval",
    "dino":   "fedDINO_all_eval",
    "tips":   "fedTIPS_all_eval",
    "resnet": "fedResNet_all_eval",
}


def infer_module(backbone_tag: str) -> str:
    prefix = backbone_tag.split("_", 1)[0]
    if prefix not in BACKBONE_MODULES:
        sys.exit(f"Unknown backbone family in tag '{backbone_tag}'. "
                 f"Expected prefix in {sorted(BACKBONE_MODULES)}.")
    return BACKBONE_MODULES[prefix]


def build_eval_trainer(ckpt: dict, eval_datasets: list[str], tiny_root: str,
                       celeba_root: str | None = None,
                       cpu: bool = False):
    """
    Build a minimal FedSimCLRTrainer that has just enough state to call
    `_run_evaluations`: a global_model with the loaded projector, the cfg, and
    the device. We do NOT instantiate FL clients or partition data.
    """
    saved_cfg = ckpt["config"]   # dict of stringified values
    backbone_tag = saved_cfg.get("backbone_tag", "")
    module_name = infer_module(backbone_tag)
    mod = importlib.import_module(module_name)

    # Reconstruct an args-like config object.
    def _coerce(v):
        if v in ("None", ""):
            return None
        if v == "True":
            return True
        if v == "False":
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    cfg = SimpleNamespace(**{k: _coerce(v) for k, v in saved_cfg.items()})
    # Override only the fields relevant to eval.
    cfg.eval_datasets = eval_datasets
    cfg.tiny_root = tiny_root
    if celeba_root is not None:
        cfg.celeba_root = celeba_root
    cfg.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    cfg.save_test_embeddings = False
    cfg.verbose = getattr(cfg, "verbose", False)
    cfg.progress_every = getattr(cfg, "progress_every", 20) or 20
    cfg.linear_lr = getattr(cfg, "linear_lr", 1e-3) or 1e-3
    cfg.linear_wd = getattr(cfg, "linear_wd", 1e-4) or 1e-4
    cfg.linear_bs = getattr(cfg, "linear_bs", 256) or 256
    cfg.linear_epochs = getattr(cfg, "linear_epochs", 50) or 50
    cfg.linear_patience = getattr(cfg, "linear_patience", 10) or 10
    cfg.batch_size = getattr(cfg, "batch_size", 128) or 128
    cfg.projection_dim = getattr(cfg, "projection_dim", 128) or 128
    cfg.normalization = getattr(cfg, "normalization", "imagenet") or "imagenet"

    print(f"Backbone tag: {backbone_tag}  (module: {module_name})")
    print(f"Eval datasets: {eval_datasets}")
    print(f"Device: {cfg.device}")

    # Build backbone + projector matching what was trained.
    SharedBB = getattr(mod, [name for name in dir(mod) if name.startswith("Shared") and name.endswith("Backbone")][0])
    UnsupervisedModel = getattr(mod, [name for name in dir(mod) if name.startswith("Unsupervised")][0])

    shared = SharedBB.get_instance(cfg)
    global_model = UnsupervisedModel(shared, projection_dim=cfg.projection_dim).to(cfg.device)
    global_model.projector.load_state_dict(ckpt["projector"])
    global_model.eval()
    print(f"Loaded projector from checkpoint (rounds_completed={ckpt.get('rounds_completed', '?')}).")

    # Construct a minimal trainer-like object that has the methods we need.
    # We can't just bind methods because _run_one_eval reads self.cfg, self.global_model, self.device.
    Trainer = mod.FedSimCLRTrainer
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.device = cfg.device
    trainer.shared = shared
    trainer.global_model = global_model
    return trainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to projector .pt")
    p.add_argument("--json", required=True,
                   help="Path to the original run's JSON; results will be appended.")
    p.add_argument("--eval_datasets", required=True,
                   help="Comma-separated list of eval datasets to add.")
    p.add_argument("--tiny_root", default="/home/daniela/mine/fedDINO/data/tiny_in_view",
                   help="Path to Tiny-ImageNet view root (only used if 'tiny' in eval list).")
    p.add_argument("--celeba_root", default=None,
                   help="Path to CelebA root (only used if 'celeba' in eval list).")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    if not ckpt_path.is_file():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    # If JSON doesn't exist (e.g. original run crashed during eval), start a
    # fresh one populated with the config from the checkpoint. Useful for
    # backfilling eval columns onto checkpoints that have no JSON yet.
    if not json_path.is_file():
        print(f"JSON not found at {json_path}; creating a fresh one from the checkpoint config.")
        json_path.parent.mkdir(parents=True, exist_ok=True)

    eval_datasets = [s.strip() for s in args.eval_datasets.split(",") if s.strip()]

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    trainer = build_eval_trainer(
        ckpt, eval_datasets=eval_datasets,
        tiny_root=args.tiny_root, celeba_root=args.celeba_root, cpu=args.cpu,
    )

    # Load existing JSON if present, else start a fresh one.
    if json_path.is_file():
        with open(json_path) as f:
            results = json.load(f)
    else:
        results = {
            "config": ckpt.get("config", {}),
            "checkpoint_path": str(ckpt_path),
            "note": "Created by eval_only.py because original run had no JSON (likely crashed during eval).",
            "eval_results": [],
            "unsup_results": [],
        }

    # Ensure the lists exist
    results.setdefault("eval_results", [])
    results.setdefault("unsup_results", [])
    n_eval_before = len(results["eval_results"])
    n_unsup_before = len(results["unsup_results"])

    # Skip eval datasets that are already present in the JSON (covers both
    # eval_results and unsup_results). Avoids double-evaluating after the
    # log-recovery step or when this script is re-run on a partial JSON.
    already_done = {entry.get("eval_dataset")
                    for entry in results["eval_results"]
                    if entry.get("eval_dataset")}
    already_done |= {entry.get("eval_dataset")
                     for entry in results["unsup_results"]
                     if entry.get("eval_dataset")}
    to_run = [d for d in eval_datasets if d not in already_done]
    skipped = [d for d in eval_datasets if d in already_done]
    if skipped:
        print(f"Skipping (already in JSON): {skipped}")
    if not to_run:
        print("All requested eval datasets already present; nothing to do.")
        return
    print(f"Will evaluate: {to_run}")
    trainer.cfg.eval_datasets = to_run

    # Use the original number of rounds (or fallback) as the "round" tag for these eval rows.
    round_num = ckpt.get("rounds_completed", getattr(trainer.cfg, "rounds", 0))
    trainer._run_evaluations(results, round_num)

    added_eval = len(results["eval_results"]) - n_eval_before
    added_unsup = len(results["unsup_results"]) - n_unsup_before
    print(f"\nAdded {added_eval} eval_results rows and {added_unsup} unsup_results rows.")

    # Backup the original JSON before overwriting (only if it's on disk).
    backup = json_path.with_suffix(json_path.suffix + ".bak")
    if json_path.is_file() and not backup.exists():
        os.rename(json_path, backup)
        print(f"Backup of original: {backup}")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated JSON: {json_path}")


if __name__ == "__main__":
    main()
