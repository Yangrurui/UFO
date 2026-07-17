#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# Adam SP Training Script for UFO / BFM-Zero
# ============================================================
# Usage:
#   # Smoke test (verify environment)
#   bash scripts/train_adam_sp.sh smoke
#
#   # Short gate (50K steps, with tracking eval)
#   bash scripts/train_adam_sp.sh gate
#
#   # Full 8-GPU training
#   bash scripts/train_adam_sp.sh train
#
#   # Custom
#   bash scripts/train_adam_sp.sh train --num-envs 512 --wandb-run-name my_run
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --------------- defaults ---------------
AGENT="${AGENT:-fb}"
ROBOT_CONFIG="${ROBOT_CONFIG:-configs/robots/adam_sp_29dof.yaml}"
DATA_PATH="${DATA_PATH:-humanoidverse/data/adam_sp_motion_train.pkl}"
GPU_IDS="${GPU_IDS:-single}"
WORK_DIR="${WORK_DIR:-runs/ufo_adam_sp}"
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-192000000}"
BUFFER_SIZE="${BUFFER_SIZE:-5120000}"
UPDATE_Z_EVERY="${UPDATE_Z_EVERY:-100}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ufo_fb_adam_sp}"
USE_WANDB="${USE_WANDB:-false}"
# ----------------------------------------

cd "$PROJECT_DIR"

die() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

# ---- mode selection ----
MODE="${1:-}"
shift 2>/dev/null || true
EXTRA_ARGS=("$@")

case "$MODE" in
  smoke)
    info "Adam SP smoke test (16 envs, 2048 steps)"
    ./run_train.sh \
      --agent "$AGENT" \
      --robot-config "$ROBOT_CONFIG" \
      --data-path "$DATA_PATH" \
      --gpu-ids single \
      --smoke \
      --buffer-size 102400 \
      --work-dir /tmp/ufo_smoke_adam_sp \
      "${EXTRA_ARGS[@]}"
    ;;

  gate)
    info "Adam SP gate training (64 envs, 50K steps)"
    ./run_train.sh \
      --agent "$AGENT" \
      --robot-config "$ROBOT_CONFIG" \
      --data-path "$DATA_PATH" \
      --gpu-ids single \
      --num-envs 64 \
      --num-env-steps 50000 \
      --buffer-size 256000 \
      --update-z-every-step "$UPDATE_Z_EVERY" \
      --work-dir /tmp/ufo_gate_adam_sp \
      "${EXTRA_ARGS[@]}"
    ;;

  train)
    if [ "$GPU_IDS" = "all" ]; then
      info "Adam SP full training (8-GPU, ${NUM_ENVS} envs/GPU, ${NUM_ENV_STEPS} steps)"
      info "CUDA_VISIBLE_DEVICES will be auto-detected"
    else
      info "Adam SP training (GPU: ${GPU_IDS}, ${NUM_ENVS} envs/GPU, ${NUM_ENV_STEPS} steps)"
    fi

    WANDB_FLAGS=()
    if [ "$USE_WANDB" = "true" ]; then
      WANDB_FLAGS=(--use-wandb --wandb-run-name "$WANDB_RUN_NAME")
    fi

    ./run_train.sh \
      --agent "$AGENT" \
      --robot-config "$ROBOT_CONFIG" \
      --data-path "$DATA_PATH" \
      --gpu-ids "$GPU_IDS" \
      --num-envs "$NUM_ENVS" \
      --num-env-steps "$NUM_ENV_STEPS" \
      --buffer-size "$BUFFER_SIZE" \
      --update-z-every-step "$UPDATE_Z_EVERY" \
      --work-dir "$WORK_DIR" \
      "${WANDB_FLAGS[@]}" \
      "${EXTRA_ARGS[@]}"
    ;;

  *)
    cat <<USAGE
Adam SP Training Script
=======================

Usage: bash scripts/train_adam_sp.sh <mode> [extra args]

Modes:
  smoke     Quick smoke test (16 envs, 2048 steps, no eval)
  gate      Short gate training (64 envs, 50K steps, with eval)
  train     Full training (configurable via env vars)

Environment variables (for 'train' mode):
  AGENT              fb (default) | tech
  GPU_IDS            single (default) | all | 0,1,2,3
  NUM_ENVS           1024 (default) — per GPU
  NUM_ENV_STEPS      192000000 (default) — global steps
  BUFFER_SIZE        5120000 (default) — per GPU
  WORK_DIR           runs/ufo_adam_sp (default)
  DATA_PATH          humanoidverse/data/adam_sp_motion_train.pkl (default)
  USE_WANDB          false (default) — set to 'true' to enable
  WANDB_RUN_NAME     ufo_fb_adam_sp (default)

Examples:
  bash scripts/train_adam_sp.sh smoke
  bash scripts/train_adam_sp.sh gate
  bash scripts/train_adam_sp.sh train
  USE_WANDB=true GPU_IDS=all bash scripts/train_adam_sp.sh train
  bash scripts/train_adam_sp.sh train --clip-grad-norm 1.0 --lr-scale 0.5
USAGE
    exit 1
    ;;
esac
