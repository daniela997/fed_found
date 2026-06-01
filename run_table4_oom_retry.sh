#!/usr/bin/env bash
# Re-run the 7 Table 4 cells that OOM-crashed during the WCO grid.
#
# Failures (all OOM on GPU 0, scheduler race where 2/GPU concurrency hit the
# 24 GB ceiling when the previous cell's memory was still draining):
#   - clip_b16_2000C_5pct  seed {1, 17, 27}    (3 cells)
#   - dino_b16_2000C_1pct  seed 17             (1 cell)
#   - dino_b16_2000C_5pct  seed {1, 27}        (2 cells)
#   - resnet18_2000C_5pct  seed 17             (1 cell)
#
# Schedule: 1 job per GPU (no concurrency), strictly sequential per GPU.
# Round-robin assignment to GPU 0 / GPU 1.

set -uo pipefail

PY=/scratch/daniela/miniconda3/envs/fedclip/bin/python

TRAIN_DATASET=wco_l4_ifcb_pml
EVAL_DATASETS=wco_l4_ifcb_pml,syke_ifcb,syke_zooscan,plankto_share,daplankton_lab_ifcb,daplankton_lab_cs,daplankton_lab_fc

LOG_DIR=fl_results/table4_plankton_logs
mkdir -p "$LOG_DIR"
mkdir -p fl_results/checkpoints

N_GPUS=${N_GPUS:-2}

declare -A BB_SCRIPT=(
  [clip_b16]="/home/daniela/mine/fedDINO/fedCLIP_all_eval.py"
  [dino_b16]="/home/daniela/mine/fedDINO/fedDINO_all_eval.py"
  [resnet18]="/home/daniela/mine/fedDINO/fedResNet_all_eval.py"
)
declare -A BB_ARGS=(
  [clip_b16]=""
  [dino_b16]="--dino_model vit_base_patch16_dinov3"
  [resnet18]="--resnet_model resnet18"
)

# Cells to re-run (backbone, num_clients, client_frac, label, seed).
JOBS=(
  "clip_b16 2000 0.05 2000C_5pct 1"
  "clip_b16 2000 0.05 2000C_5pct 17"
  "clip_b16 2000 0.05 2000C_5pct 27"
  "dino_b16 2000 0.01 2000C_1pct 17"
  "dino_b16 2000 0.05 2000C_5pct 1"
  "dino_b16 2000 0.05 2000C_5pct 27"
  "resnet18 2000 0.05 2000C_5pct 17"
)

run_one () {
  local gpu=$1; local tag=$2; local num_clients=$3; local client_frac=$4; local label=$5; local seed=$6
  local script="${BB_SCRIPT[$tag]}"
  local model_args="${BB_ARGS[$tag]}"
  local run_tag="${tag}_${label}_seed${seed}"
  local logf="${LOG_DIR}/${run_tag}.log"

  # Three cases:
  #   1. JSON exists -> SKIP (cell already complete).
  #   2. Checkpoint exists but JSON doesn't -> training succeeded, eval crashed.
  #      Re-run only the eval phase via eval_only.py (much faster).
  #   3. Neither exists -> full FL training + eval.
  local existing_ckpt
  existing_ckpt=$(ls -t fl_results/checkpoints/projector_${tag}_${TRAIN_DATASET}_${run_tag}_*.pt 2>/dev/null | head -1)
  local existing_json
  if [ -n "${existing_ckpt}" ]; then
    local ckpt_basename
    ckpt_basename=$(basename "${existing_ckpt}" .pt)
    local json_path="fl_results/fed_unsup_simclr_${ckpt_basename#projector_}.json"
    if [ -f "${json_path}" ]; then
      echo "[GPU ${gpu}] SKIP ${run_tag} -- already complete"
      return
    fi
    echo "[GPU ${gpu}] EVAL-ONLY ${run_tag} (ckpt exists, JSON missing)"
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

# 1 job per GPU. Round-robin assignment, but ensure that a GPU's previous job
# has fully exited before its next job starts. Track one pid per gpu.

declare -A PID_PER_GPU=()

echo "Table 4 OOM retry: ${#JOBS[@]} cells at 1 job/GPU"
START=$(date +%s)

i=0
for jobspec in "${JOBS[@]}"; do
  read -r tag nc cf lab s <<< "$jobspec"
  gpu=$(( i % N_GPUS ))
  prev_pid="${PID_PER_GPU[$gpu]:-}"
  if [ -n "$prev_pid" ]; then
    wait "$prev_pid" || true
  fi
  run_one "$gpu" "$tag" "$nc" "$cf" "$lab" "$s" &
  PID_PER_GPU[$gpu]=$!
  i=$(( i + 1 ))
done

for pid in "${PID_PER_GPU[@]}"; do
  wait "$pid" || true
done

END=$(date +%s)
echo
echo "All ${#JOBS[@]} retry runs complete in $(( END - START ))s."
echo "Aggregate with:"
echo "  $PY /home/daniela/mine/fedDINO/aggregate_table4_plankton.py"
