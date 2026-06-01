#!/usr/bin/env bash
# Driver for Table 4: federated SimCLR trained on WCO L4 IFCB, evaluated on
# the 7 plankton datasets we uploaded to HuggingFace.
#
# Mirrors run_table3_all_backbones.sh but with:
#   - training dataset = wco_l4_ifcb_pml (Western Channel IFCB, ~200 classes,
#                        ~90.7k images, 60/20/20 seed 24)
#   - eval datasets    = wco_l4_ifcb_pml (in-domain) + 6 cross-domain plankton
#                        sets (syke_ifcb, syke_zooscan, plankto_share,
#                        daplankton_lab_{cs,fc,ifcb})
#
# Grid: 5 backbones x {200, 2000} clients x {0.01, 0.05} participation x 3 seeds
#       = 60 runs.
# Hyperparams: 20 rounds, 5 local epochs, batch 128, alpha=0.3 unbalanced Dirichlet.
# Seeds: {1, 17, 27}.
#
# Concurrency: same two-phase scheduling as Table 3 (200C cells 2/GPU, 2000C 1/GPU).

set -uo pipefail

PY=/scratch/daniela/miniconda3/envs/fedclip/bin/python

TRAIN_DATASET=wco_l4_ifcb_pml
EVAL_DATASETS=wco_l4_ifcb_pml,syke_ifcb,syke_zooscan,plankto_share,daplankton_lab_ifcb,daplankton_lab_cs,daplankton_lab_fc

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
MODEL_ARGS=(
  ""
  "--dino_model vit_small_patch16_dinov3"
  "--dino_model vit_base_patch16_dinov3"
  ""
  "--resnet_model resnet18"
)

SEEDS=(1 17 27)
SMALL_CELLS=(
  "200  0.01 200C_1pct"
  "200  0.05 200C_5pct"
)
LARGE_CELLS=(
  "2000 0.01 2000C_1pct"
  "2000 0.05 2000C_5pct"
)

LOG_DIR=fl_results/table4_plankton_logs
mkdir -p "$LOG_DIR"
mkdir -p fl_results/checkpoints

N_GPUS=${N_GPUS:-2}
JOBS_PER_GPU_SMALL=${JOBS_PER_GPU_SMALL:-2}
JOBS_PER_GPU_PHASE2A=${JOBS_PER_GPU_PHASE2A:-2}
JOBS_PER_GPU_PHASE2B=${JOBS_PER_GPU_PHASE2B:-1}
PHASE2B_BB_TAGS="tips_b14"

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
# whitespace-separated list of allowed backbone tags.
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

  # Reuse an existing projector checkpoint if one exists for this (tag, run_tag).
  # Filenames follow:
  #   projector_<tag>_syke_ifcb_<run_tag>_<YYYYMMDD>_<HHMMSS>.pt
  local existing_ckpt
  existing_ckpt=$(ls -t fl_results/checkpoints/projector_${tag}_${TRAIN_DATASET}_${run_tag}_*.pt 2>/dev/null | head -1)

  if [ -n "${existing_ckpt}" ] && [ -f "${existing_ckpt}" ]; then
    local ckpt_basename
    ckpt_basename=$(basename "${existing_ckpt}" .pt)
    local json_basename="fed_unsup_simclr_${ckpt_basename#projector_}.json"
    local json_path="fl_results/${json_basename}"
    echo "[GPU ${gpu}] reusing checkpoint for ${run_tag} -> eval_only.py"
    CUDA_VISIBLE_DEVICES=${gpu} "$PY" /home/daniela/mine/fedDINO/eval_only.py \
      --checkpoint "${existing_ckpt}" \
      --json "${json_path}" \
      --eval_datasets "${EVAL_DATASETS}" \
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
    --dataset "${TRAIN_DATASET}" --eval_datasets "${EVAL_DATASETS}" \
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

echo "Table 4 (plankton) three-phase schedule across ${N_GPUS} GPUs:"
echo "  Train dataset: ${TRAIN_DATASET}"
echo "  Eval datasets: ${EVAL_DATASETS}"
echo "  Phase 1  (200C all backbones):                       ${#SMALL_JOBS[@]} runs at ${JOBS_PER_GPU_SMALL} jobs/GPU"
echo "  Phase 2a (2000C [${PHASE2A_BB_TAGS}]): ${#PHASE2A_JOBS[@]} runs at ${JOBS_PER_GPU_PHASE2A} jobs/GPU"
echo "  Phase 2b (2000C [${PHASE2B_BB_TAGS}]):                       ${#PHASE2B_JOBS[@]} runs at ${JOBS_PER_GPU_PHASE2B} jobs/GPU"
echo "  Total: ${TOTAL} runs"
START=$(date +%s)

run_phase "200C (all backbones)"        "$JOBS_PER_GPU_SMALL"    "${SMALL_JOBS[@]}"
run_phase "2000C (everything but TIPS)" "$JOBS_PER_GPU_PHASE2A"  "${PHASE2A_JOBS[@]}"
run_phase "2000C (TIPS)"                "$JOBS_PER_GPU_PHASE2B"  "${PHASE2B_JOBS[@]}"

END=$(date +%s)
echo
echo "All ${TOTAL} runs complete in $(( END - START ))s."
echo "Aggregate with:"
echo "  $PY /home/daniela/mine/fedDINO/aggregate_table4_plankton.py"
