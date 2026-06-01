#!/usr/bin/env bash
# Driver for all backbones (CLIP-B/16, DINOv3-S/16, DINOv3-B/16, TIPS-B/14, ResNet-50).
# Grid: 5 backbones x {200, 2000} clients x {0.01, 0.05} participation x 3 seeds
#       = 60 runs.
# Hyperparams: 20 rounds, 5 local epochs, batch 128, alpha=0.3 unbalanced Dirichlet.
# Seeds: {1, 17, 27}.
#
# Each backbone is identified by a short tag that's used in result filenames
# and aggregator regexes. Currently:
#   clip_b16  ->  OpenAI CLIP ViT-B/16 (quickgelu)
#   dino_s16  ->  DINOv3 ViT-S/16 (timm)
#   dino_b16  ->  DINOv3 ViT-B/16 (timm)
#   tips_b14  ->  TIPSv2 B/14 (HuggingFace google/tipsv2-b14)
#   resnet18  ->  torchvision ResNet-18 IMAGENET1K_V1 (supervised CNN baseline)
#                 (switched from ResNet-50: 2000C x 5% OOMs on 24GB even at 1 job/GPU
#                  due to wide 2048-D projector copies and large early conv activations.
#                  ResNet-18's 512-D features and ~4x fewer parameters fit comfortably.)
#
# Eval datasets (every run): cifar10, cifar100, tiny (in-domain) + whoi_plankton,
# eurosat, bone_marrow, wikiart_style (out-of-domain). Projector checkpoints are
# saved to fl_results/checkpoints/ so OOD evals can also be re-run later
# without re-training (via eval_only.py).
#
# Concurrency: three-phase scheduling tuned to GPU memory pressure per cell:
#   Phase 1  - 200C cells (all backbones)                            : 2 jobs/GPU = 4 concurrent
#   Phase 2a - 2000C cells for resnet18 + dino_s16 + clip_b16 + dino_b16 : 2 jobs/GPU = 4 concurrent
#   Phase 2b - 2000C cells for tips_b14                              : 1 job/GPU  = 2 concurrent
# At 2000 clients, ResNet-18 (with the rest) just barely fits 2/GPU on a 24 GB
# card during the eval phase. TIPS-B/14 was historically the worst memory case
# (large model + 768-D features + image-text training stack carried over), so
# it gets its own 1/GPU phase as a safety margin.
# Note: 3/GPU at 2000C OOMs even for ResNet-18 because the three processes
# collectively consume ~23.5 GB at training peak, leaving zero headroom.
#
# Override defaults:
#   JOBS_PER_GPU_SMALL=N JOBS_PER_GPU_PHASE2A=M JOBS_PER_GPU_PHASE2B=K N_GPUS=G \\
#     ./run_table3_all_backbones.sh

set -uo pipefail

PY=/scratch/daniela/miniconda3/envs/fedclip/bin/python
TINY_ROOT=/home/daniela/mine/fedDINO/data/tiny_in_view
EVAL_DATASETS=cifar10,cifar100,tiny,whoi_plankton,eurosat,bone_marrow,wikiart_style

# (backbone_tag, script_path, extra_args_for_backbone_specifics)
# We declare the 4 backbones as parallel arrays since bash assoc arrays + tuples
# get awkward. Index alignment is: TAGS[i] uses SCRIPTS[i] with MODEL_ARGS[i].
TAGS=(
  clip_b16
  dino_s16
  dino_b16
  tips_b14
  resnet18
)
SCRIPTS=(
  /home/daniela/mine/fedDINO/fedCLIP_all_eval.py
  /home/daniela/mine/fedDINO/fedDINO_all_eval.py
  /home/daniela/mine/fedDINO/fedDINO_all_eval.py
  /home/daniela/mine/fedDINO/fedTIPS_all_eval.py
  /home/daniela/mine/fedDINO/fedResNet_all_eval.py
)
# Per-backbone extra args (model selection, mostly relevant for DINO).
MODEL_ARGS=(
  ""
  "--dino_model vit_small_patch16_dinov3"
  "--dino_model vit_base_patch16_dinov3"
  ""
  "--resnet_model resnet18"
)

SEEDS=(1 17 27)
# Split cells into "small" (200C) and "large" (2000C) for per-phase concurrency.
SMALL_CELLS=(
  "200  0.01 200C_1pct"
  "200  0.05 200C_5pct"
)
LARGE_CELLS=(
  "2000 0.01 2000C_1pct"
  "2000 0.05 2000C_5pct"
)

LOG_DIR=fl_results/table3_all_logs
mkdir -p "$LOG_DIR"
mkdir -p fl_results/checkpoints

N_GPUS=${N_GPUS:-2}
# Phase 1 (200C, all backbones).
JOBS_PER_GPU_SMALL=${JOBS_PER_GPU_SMALL:-2}
# Phase 2a (2000C, everything except TIPS).
JOBS_PER_GPU_PHASE2A=${JOBS_PER_GPU_PHASE2A:-2}
# Phase 2b (2000C, TIPS only).
JOBS_PER_GPU_PHASE2B=${JOBS_PER_GPU_PHASE2B:-1}
# Backbones routed to Phase 2b (1 job/GPU). TIPS-B/14 has the largest peak.
PHASE2B_BB_TAGS="tips_b14"

# Backwards-compat: legacy JOBS_PER_GPU overrides everything.
if [ -n "${JOBS_PER_GPU:-}" ]; then
  JOBS_PER_GPU_SMALL="$JOBS_PER_GPU"
  JOBS_PER_GPU_PHASE2A="$JOBS_PER_GPU"
  JOBS_PER_GPU_PHASE2B="$JOBS_PER_GPU"
fi
if [ -n "${JOBS_PER_GPU_LARGE:-}" ]; then
  JOBS_PER_GPU_PHASE2A="$JOBS_PER_GPU_LARGE"
  JOBS_PER_GPU_PHASE2B="$JOBS_PER_GPU_LARGE"
fi

# Build a job list from a given set of cells, optionally filtered to a
# whitespace-separated list of allowed backbone tags. Encodes backbone index
# so run_one can look up SCRIPTS/MODEL_ARGS.
#   $1 = name-ref to cells array
#   $2 = name-ref to output jobs array
#   $3 = (optional) space-separated allowed backbone tags. If empty, all backbones.
build_jobs () {
  local -n cells_ref=$1
  local -n out_ref=$2
  local allowed="${3:-}"
  out_ref=()
  for i_bb in "${!TAGS[@]}"; do
    local tag="${TAGS[$i_bb]}"
    if [ -n "${allowed}" ] && [[ " ${allowed} " != *" ${tag} "* ]]; then
      continue
    fi
    for cell in "${cells_ref[@]}"; do
      for seed in "${SEEDS[@]}"; do
        out_ref+=("${i_bb} ${cell} ${seed}")
      done
    done
  done
}

run_one () {
  local gpu=$1; local i_bb=$2; local num_clients=$3; local client_frac=$4; local label=$5; local seed=$6
  local tag="${TAGS[$i_bb]}"
  local script="${SCRIPTS[$i_bb]}"
  local model_args="${MODEL_ARGS[$i_bb]}"
  local run_tag="${tag}_${label}_seed${seed}"
  local logf="${LOG_DIR}/${run_tag}.log"

  # If a checkpoint for this (tag, run_tag) already exists, skip FL training
  # and run eval_only.py on the existing checkpoint to produce/extend the JSON.
  # Glob matches the timestamped filename:
  #   projector_<tag>_cifar10_<run_tag>_<YYYYMMDD>_<HHMMSS>.pt
  local existing_ckpt
  existing_ckpt=$(ls -t fl_results/checkpoints/projector_${tag}_cifar10_${run_tag}_*.pt 2>/dev/null | head -1)

  if [ -n "${existing_ckpt}" ] && [ -f "${existing_ckpt}" ]; then
    # Derive matching JSON path (replace `projector_` prefix and `.pt` suffix).
    local ckpt_basename
    ckpt_basename=$(basename "${existing_ckpt}" .pt)
    local json_basename="fed_unsup_simclr_${ckpt_basename#projector_}.json"
    local json_path="fl_results/${json_basename}"
    echo "[GPU ${gpu}] reusing checkpoint for ${run_tag} -> eval_only.py"
    CUDA_VISIBLE_DEVICES=${gpu} "$PY" /home/daniela/mine/fedDINO/eval_only.py \
      --checkpoint "${existing_ckpt}" \
      --json "${json_path}" \
      --eval_datasets "${EVAL_DATASETS}" \
      --tiny_root "${TINY_ROOT}" \
      > "${logf}" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "[GPU ${gpu}] done (eval-only) ${run_tag}"
    else
      echo "[GPU ${gpu}] FAILED eval_only ${run_tag} (exit ${rc}); see ${logf}"
    fi
    return
  fi

  echo "[GPU ${gpu}] starting ${run_tag}"
  CUDA_VISIBLE_DEVICES=${gpu} "$PY" "$script" \
    --dataset cifar10 --eval_datasets "${EVAL_DATASETS}" \
    --tiny_root "${TINY_ROOT}" \
    --backbone_tag "${tag}" ${model_args} \
    --num_clients "${num_clients}" --client_frac "${client_frac}" \
    --rounds 20 --local_epochs 5 --batch_size 128 \
    --seed "${seed}" --run_name "${run_tag}" \
    > "${logf}" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[GPU ${gpu}] done ${run_tag}"
  else
    echo "[GPU ${gpu}] FAILED ${run_tag} (exit ${rc}); see ${logf}"
  fi
}

# Run one phase: schedule all jobs in `phase_jobs` at `jobs_per_gpu` jobs/GPU,
# round-robin GPU assignment.
run_phase () {
  local phase_name=$1
  local jobs_per_gpu=$2
  shift 2
  local phase_jobs=("$@")
  local max_concurrency=$(( N_GPUS * jobs_per_gpu ))
  echo
  echo "=== Phase '${phase_name}': ${#phase_jobs[@]} runs, ${jobs_per_gpu} jobs/GPU x ${N_GPUS} GPUs = ${max_concurrency} concurrent ==="
  local running=0
  for i in "${!phase_jobs[@]}"; do
    read -r i_bb nc cf lab s <<< "${phase_jobs[$i]}"
    local gpu=$(( i % N_GPUS ))
    if [ "$running" -ge "$max_concurrency" ]; then
      wait -n || true
      running=$(( running - 1 ))
    fi
    run_one "$gpu" "$i_bb" "$nc" "$cf" "$lab" "$s" &
    running=$(( running + 1 ))
  done
  wait
}

# Phase 2a backbones = TAGS minus PHASE2B_BB_TAGS.
PHASE2A_BB_TAGS=""
for tag in "${TAGS[@]}"; do
  if [[ " ${PHASE2B_BB_TAGS} " != *" ${tag} "* ]]; then
    PHASE2A_BB_TAGS+="${tag} "
  fi
done

build_jobs SMALL_CELLS SMALL_JOBS
build_jobs LARGE_CELLS PHASE2A_JOBS "${PHASE2A_BB_TAGS}"
build_jobs LARGE_CELLS PHASE2B_JOBS "${PHASE2B_BB_TAGS}"
TOTAL=$(( ${#SMALL_JOBS[@]} + ${#PHASE2A_JOBS[@]} + ${#PHASE2B_JOBS[@]} ))

echo "Three-phase schedule across ${N_GPUS} GPUs:"
echo "  Phase 1  (200C all backbones):  ${#SMALL_JOBS[@]} runs at ${JOBS_PER_GPU_SMALL} jobs/GPU"
echo "  Phase 2a (2000C [${PHASE2A_BB_TAGS}]): ${#PHASE2A_JOBS[@]} runs at ${JOBS_PER_GPU_PHASE2A} jobs/GPU"
echo "  Phase 2b (2000C [${PHASE2B_BB_TAGS}]):                       ${#PHASE2B_JOBS[@]} runs at ${JOBS_PER_GPU_PHASE2B} jobs/GPU"
echo "  Total: ${TOTAL} runs"
START=$(date +%s)

run_phase "200C (all backbones)"     "$JOBS_PER_GPU_SMALL"     "${SMALL_JOBS[@]}"
run_phase "2000C (everything but TIPS)" "$JOBS_PER_GPU_PHASE2A" "${PHASE2A_JOBS[@]}"
run_phase "2000C (TIPS)"             "$JOBS_PER_GPU_PHASE2B"   "${PHASE2B_JOBS[@]}"

END=$(date +%s)
echo
echo "All ${TOTAL} runs complete in $(( END - START ))s."
echo "Aggregate with:"
echo "  $PY /home/daniela/mine/fedDINO/aggregate_table3_compare.py"
