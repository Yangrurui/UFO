#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# Adam SP Inference Script for UFO / BFM-Zero
# ============================================================
# Usage:
#   # Tracking inference
#   bash scripts/infer_adam_sp.sh tracking \
#     --model-folder runs/ufo_adam_sp \
#     --motion-list 0 1 2
#
#   # Goal inference
#   bash scripts/infer_adam_sp.sh goal \
#     --model-folder runs/ufo_adam_sp \
#     --goal-json /path/to/adam_sp_goals.json
#
#   # Reward inference
#   bash scripts/infer_adam_sp.sh reward \
#     --model-folder runs/ufo_adam_sp \
#     --tasks move-ego-0-0 move-ego-0-0.7
#
#   # Export ONNX only
#   bash scripts/infer_adam_sp.sh onnx \
#     --model-folder runs/ufo_adam_sp
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --------------- defaults ---------------
ROBOT_CONFIG="${ROBOT_CONFIG:-configs/robots/adam_sp_29dof.yaml}"
DATA_PATH="${DATA_PATH:-humanoidverse/data/adam_sp_motion_infer.pkl}"
DEVICE="${DEVICE:-cuda:0}"
HEADLESS="${HEADLESS:-true}"
SAVE_MP4="${SAVE_MP4:-}"           # set to 'true' to enable
EXPORT_ONNX="${EXPORT_ONNX:-}"     # set to 'true' to enable
MOTION_LIST="${MOTION_LIST:-0}"
FPS="${FPS:-30}"
CAMERA_DISTANCE="${CAMERA_DISTANCE:-1.734}"
CAMERA_AZIMUTH="${CAMERA_AZIMUTH:--1.135}"
CAMERA_ELEVATION="${CAMERA_ELEVATION:-0.35}"
MAX_EPISODE_LENGTH_S="${MAX_EPISODE_LENGTH_S:-30}"
# ----------------------------------------

cd "$PROJECT_DIR"

die() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

MODE="${1:-}"
shift 2>/dev/null || true

# ---- resolve model folder ----
MODEL_FOLDER=""
for arg in "$@"; do
  case "$arg" in
    --model-folder) ;;  # skip, value is next
    --model-folder=*) MODEL_FOLDER="${arg#--model-folder=}" ;;
    *)
      if [ "${prev:-}" = "--model-folder" ]; then
        MODEL_FOLDER="$arg"
      fi
      ;;
  esac
  prev="$arg"
done

# ---- build common flags ----
COMMON_FLAGS=(
  --robot-config "$ROBOT_CONFIG"
)

if [ -n "$MODEL_FOLDER" ]; then
  COMMON_FLAGS+=(--model-folder "$MODEL_FOLDER")
fi

MP4_FLAGS=()
if [ "$SAVE_MP4" = "true" ]; then
  MP4_FLAGS=(--save-mp4)
fi

ONNX_FLAGS=()
if [ "$EXPORT_ONNX" = "true" ]; then
  ONNX_FLAGS=(--export-onnx true)
fi

# ---- mode dispatch ----
case "$MODE" in
  tracking)
    info "Adam SP tracking inference"
    uv run python -m humanoidverse.tracking_inference \
      "${COMMON_FLAGS[@]}" \
      --data-path "$DATA_PATH" \
      --device "$DEVICE" \
      --headless \
      --disable-dr \
      --disable-obs-noise \
      --motion-list $MOTION_LIST \
      --fps "$FPS" \
      --camera-distance "$CAMERA_DISTANCE" \
      --camera-azimuth "$CAMERA_AZIMUTH" \
      --camera-elevation "$CAMERA_ELEVATION" \
      --max-episode-length-s "$MAX_EPISODE_LENGTH_S" \
      "${MP4_FLAGS[@]}" \
      "${ONNX_FLAGS[@]}" \
      "$@"
    ;;

  goal)
    info "Adam SP goal inference"
    if ! echo "$*" | grep -q "\--goal-json"; then
      cat >&2 <<WARN
========================================
  WARNING: No --goal-json provided.
  Adam SP is 29-DOF — goal_inference.py WILL silently load
  G1's default goal_frames_lafan29dof.json, producing incorrect results.
  Pass --goal-json /path/to/adam_sp_goal_frames.json
========================================
WARN
    fi

    uv run python -m humanoidverse.goal_inference \
      "${COMMON_FLAGS[@]}" \
      --data-path "$DATA_PATH" \
      --device "$DEVICE" \
      --headless \
      --disable-dr \
      --disable-obs-noise \
      "${MP4_FLAGS[@]}" \
      "${ONNX_FLAGS[@]}" \
      "$@"
    ;;

  reward)
    info "Adam SP reward inference (locomotion tasks only for non-G1 robots)"
    uv run python -m humanoidverse.reward_inference \
      "${COMMON_FLAGS[@]}" \
      --device "$DEVICE" \
      --headless \
      --disable-dr \
      --disable-obs-noise \
      --buffer-rank 0 \
      --num-samples 150000 \
      --n-inferences 1 \
      "${MP4_FLAGS[@]}" \
      "${ONNX_FLAGS[@]}" \
      "$@"
    ;;

  onnx)
    info "Adam SP ONNX export"
    uv run python -m humanoidverse.tracking_inference \
      "${COMMON_FLAGS[@]}" \
      --data-path "$DATA_PATH" \
      --device "$DEVICE" \
      --headless \
      --disable-dr \
      --disable-obs-noise \
      --motion-list 0 \
      --export-onnx true \
      "$@"
    ;;

  *)
    cat <<USAGE
Adam SP Inference Script
=========================

Usage: bash scripts/infer_adam_sp.sh <mode> [flags]

Modes:
  tracking    Track reference motion sequences
  goal        Reach target poses (requires --goal-json for Adam SP)
  reward      Optimize arbitrary reward functions (locomotion only)
  onnx        Export ONNX policy + backward encoder

Common flags:
  --model-folder PATH       Path to training output dir (required)
  --robot-config PATH       Robot YAML (default: configs/robots/adam_sp_29dof.yaml)
  --data-path PATH          Motion data pkl (default: humanoidverse/data/adam_sp_motion_infer.pkl)

Environment variables:
  DEVICE              cuda:0 (default)
  MOTION_LIST         0 (default) — space-separated for multiple motions
  SAVE_MP4            set to 'true' to save videos
  EXPORT_ONNX         set to 'true' to export ONNX models
  FPS                 30 (default)
  MAX_EPISODE_LENGTH_S  30 (default)

Examples:
  # Track motion #0 with MP4
  SAVE_MP4=true bash scripts/infer_adam_sp.sh tracking \
    --model-folder runs/ufo_adam_sp --motion-list 0

  # Goal inference (REQUIRES Adam SP goal JSON)
  bash scripts/infer_adam_sp.sh goal \
    --model-folder runs/ufo_adam_sp \
    --goal-json /path/to/adam_sp_goal_frames.json

  # Export ONNX
  bash scripts/infer_adam_sp.sh onnx \
    --model-folder runs/ufo_adam_sp

  # Reward inference (locomotion tasks)
  bash scripts/infer_adam_sp.sh reward \
    --model-folder runs/ufo_adam_sp \
    --tasks move-ego-0-0 rotate-z-5-0.5
USAGE
    exit 1
    ;;
esac
