# Federated SimCLR with Frozen Vision Foundation-Model Backbones

Federated representation learning that keeps the entire backbone frozen and
federates only a lightweight projection head over five vision foundation
models: **CLIP-B/16**, **DINOv3-S/16**, **DINOv3-B/16**, **TIPSv2-B/14**, and
**ResNet-18** (a supervised CNN baseline).

The pipeline trains a SimCLR projector on top of each backbone via FedAvg
across many clients under unbalanced Dirichlet partitions, then evaluates the
learned representation with five complementary metrics (linear probe, k-NN,
K-Means cluster accuracy, prototype accuracy, silhouette score) on held-out
test sets.

## Three experimental setups

| Setup | Train | Eval | Goal |
|---|---|---|---|
| **(i) In-domain** | CIFAR-10 | CIFAR-10, CIFAR-100, Tiny-ImageNet | Federated training capacity on natural images. |
| **(ii) Cross-domain OOD** | CIFAR-10 | WHOI plankton, EuroSAT, Bone-Marrow, WikiArt | Whether the projector preserves backbone transferability. |
| **(iii) Domain-specific** | WCO L4 IFCB | WCO L4 IFCB, SYKE-IFCB, SYKE-ZooScan, PlanktoShare, DAPlankton-LAB (×3) | Federated SSL on a realistic biological-imaging task with cross-instrument transfer. |

Each setup is evaluated under four federated cells crossing **client count**
∈ {200, 2000} with **per-round participation** ∈ {1%, 5%}, three seeds, on an
unbalanced Dirichlet partition with α = 0.3.

## Repository layout

```
.
├── fed{CLIP,DINO,TIPS,ResNet}_all_eval.py   # one trainer per backbone
├── datasets_extra.py                        # HF dataset registry + adapter
├── eval_only.py                             # re-evaluate from a saved projector checkpoint
├── run_table3_all_backbones.sh              # driver: setups (i) + (ii) grid (60 cells)
├── run_table4_plankton.sh                   # driver: setup (iii) grid (60 cells)
├── run_table4_oom_retry.sh                  # driver: re-run 1/GPU for OOM-prone cells
├── aggregate_table3_compare.py              # aggregator for setups (i) + (ii)
├── aggregate_table4_plankton.py             # aggregator for setup (iii)
├── visualise.ipynb                          # notebook with figures + analysis
├── figures/                                 # generated PDF figures for the paper
└── fl_results/
    ├── table3_jsons.zip                     # all per-run JSONs for setups (i) + (ii)
    ├── table4_jsons.zip                     # all per-run JSONs for setup (iii)
    ├── table3_all_logs/                     # per-cell training logs
    └── table4_plankton_logs/                # per-cell training logs
```

The `data/` and `fl_results/checkpoints/` directories are gitignored — they
hold a ~1.1 GB dataset cache and ~230 MB of projector checkpoints respectively,
both regeneratable.

## Reproducing the experiments

### 1. Environment

```bash
conda create -n fedclip python=3.10
conda activate fedclip
pip install torch torchvision timm scikit-learn matplotlib pandas datasets huggingface_hub seaborn
```

A GPU with ≥24 GB VRAM is recommended (the 2000C cells touch the memory
ceiling for the larger backbones).

### 2. Datasets

- **CIFAR-10 / CIFAR-100**: torchvision auto-downloads on first use.
- **Tiny-ImageNet**: download from
  http://cs231n.stanford.edu/tiny-imagenet-200.zip and unpack so that
  `train/<class>/<image>.JPEG` and `val/<class>/<image>.JPEG` exist; point
  the trainers at it via `--tiny_root /path/to/tiny-imagenet-200`.
- **OOD evaluation datasets** (WHOI, EuroSAT, Bone-Marrow, WikiArt): loaded
  on demand from HuggingFace via the registry in `datasets_extra.py`.
- **Plankton datasets** (WCO L4 IFCB, SYKE-IFCB, SYKE-ZooScan, PlanktoShare,
  DAPlankton-LAB ×3): all uploaded to HuggingFace under `danielaivanova/*`;
  loaded via the same registry. The `hf_upload_configs/` YAMLs document the
  exact split protocol used when uploading.

### 3. Run a single backbone

```bash
python fedDINO_all_eval.py \
    --dataset cifar10 \
    --eval_datasets cifar10,cifar100,tiny,whoi_plankton,eurosat,bone_marrow,wikiart_style \
    --dino_model vit_base_patch16_dinov3 \
    --num_clients 200 --client_frac 0.05 \
    --rounds 20 --local_epochs 5 --batch_size 128 \
    --seed 1 --run_name dino_b16_200C_5pct_seed1
```

Outputs:
- `fl_results/fed_unsup_simclr_<tag>_<train>_<run_name>_<ts>.json` — metrics
- `fl_results/checkpoints/projector_<tag>_<train>_<run_name>_<ts>.pt` — saved projector

### 4. Run the full grid

```bash
# Setups (i) + (ii): 60 cells, ~12h on 2x A5000.
./run_table3_all_backbones.sh

# Setup (iii): 60 cells, ~14h.
./run_table4_plankton.sh
```

The drivers schedule jobs across GPUs at 2 jobs/GPU for 200-client cells
and 1–2 jobs/GPU for 2000-client cells (TIPSv2-B/14 is restricted to 1/GPU
because of its memory footprint).

If individual cells OOM (most common: clip_b16 or dino_b16 at 2000C/5%),
re-run them at 1/GPU concurrency:

```bash
./run_table4_oom_retry.sh
```

### 5. Aggregate

```bash
python aggregate_table3_compare.py        # tables for setups (i) + (ii)
python aggregate_table4_plankton.py       # tables for setup (iii)

# Optional: omit syke_ifcb (also present as plankton OOD eval) from the report
DROP_SYKE_IFCB=1 python aggregate_table4_plankton.py
```

### 6. Re-evaluate from saved checkpoints (no re-training)

```bash
python eval_only.py \
    --checkpoint fl_results/checkpoints/projector_dino_b16_cifar10_dino_b16_200C_5pct_seed1_*.pt \
    --json fl_results/<matching>.json \
    --eval_datasets cifar10,cifar100,tiny
```

Useful when adding a new eval dataset to a grid that's already trained.

## Plankton datasets on HuggingFace

The six plankton datasets used for setup (iii) are hosted on the HuggingFace
Hub and loaded on demand via `datasets_extra.py`:

| Dataset | Repo | Classes | Images |
|---|---|---|---|
| WCO L4 IFCB | `danielaivanova/wco-l4-ifcb-pml` | 200 | ~90.7k |
| SYKE-IFCB 2022 | `danielaivanova/syke-plankton-ifcb-2022` | 50 | ~63k |
| SYKE-ZooScan 2024 | `danielaivanova/syke-plankton-zooscan-2024` | 20 | ~22.7k |
| PlanktoShare | `danielaivanova/plankto-share` | 111 | ~52.8k |
| DAPlankton-LAB IFCB | `danielaivanova/daplankton-lab-ifcb` | 15 | ~16.5k |
| DAPlankton-LAB CytoSense | `danielaivanova/daplankton-lab-cs` | 15 | ~13.2k |
| DAPlankton-LAB FlowCam | `danielaivanova/daplankton-lab-fc` | 15 | ~17.8k |

All seven (including the WCO in-domain training set) use a per-class 60/20/20
train/validation/test split with random seed 24, except SYKE-ZooScan which
ships predefined splits with 5 test-only classes for open-set evaluation.

## Figures

`visualise.ipynb` produces all paper figures from the bundled JSON zips. Strip
notebook outputs before re-committing:

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace visualise.ipynb
```

Generated figures land in `figures/` (one per radial-bar setup, plus the
combined views).
