#!/usr/bin/env bash
# Unified CUDA launcher for RoboTwin LoRA or full-parameter post-training.
#
# Examples:
#   # LoRA on 1x 48G/80G GPU (no depth/video teachers)
#   MODEL_PATH=/path/to/lingbot-vla-v2-6b \
#   TOKENIZER_PATH=/path/to/Qwen3-VL-4B-Instruct \
#   DATA_LIST=assets/training_data/robotwin.txt \
#   bash scripts/train_robotwin_cuda.sh --mode lora
#
#   # Full SFT without teachers (lighter)
#   ... bash scripts/train_robotwin_cuda.sh --mode full --teacher none
#
#   # Full SFT with depth+video teachers (needs multi-GPU + teacher weights)
#   MOGE_PATH=... MORGBD_PATH=... DINO_CKPT=... DINO_CONFIG=... \
#   bash scripts/train_robotwin_cuda.sh --mode full --teacher full --gpus 4
#
# After LoRA training, merge with:
#   python scripts/merge_lora_dcp.py \
#     --checkpoint output/.../checkpoints/global_step_N \
#     --training-output output/... \
#     --base-model "$MODEL_PATH" \
#     --output output/.../merged_hf_ckpt \
#     --rank 8 --alpha 16

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="lora"
TEACHER_MODE="none"
GPU_COUNT=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/train_robotwin_cuda.sh --mode lora|full [options]

Options:
  --mode lora|full          Training mode (required via flag or MODE=)
  --teacher none|full       Teacher distillation for full mode (default: none)
  --gpus N                  Number of GPUs (default: from CUDA_VISIBLE_DEVICES or nvidia-smi)
  --config PATH             YAML config (default: configs/vla/robotwin/cuda_train.yaml)
  --output-dir PATH         Checkpoint output directory
  --data-list PATH          Multi-dataset list or single LeRobot path
  --max-steps N             Max training steps
  --micro-batch N           Per-GPU micro batch size
  --global-batch N          Global batch size
  --grad-accum N            Gradient accumulation steps
  --lora-rank N             LoRA rank (lora mode)
  --lora-alpha N            LoRA alpha (lora mode)
  -h, --help                Show this help

Environment overrides:
  MODEL_PATH, TOKENIZER_PATH, DATA_LIST, OUTPUT_DIR, MAX_STEPS,
  MICRO_BATCH_SIZE, GLOBAL_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
  LORA_RANK, LORA_ALPHA, MOGE_PATH, MORGBD_PATH, DINO_CKPT, DINO_CONFIG,
  CUDA_VISIBLE_DEVICES
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --teacher) TEACHER_MODE="$2"; shift 2 ;;
    --gpus) GPU_COUNT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --data-list) DATA_LIST="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --micro-batch) MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --global-batch) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --lora-rank) LORA_RANK="$2"; shift 2 ;;
    --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

MODE="${MODE:-lora}"
TEACHER_MODE="${TEACHER_MODE:-none}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/vla/robotwin/cuda_train.yaml}"
DATA_LIST="${DATA_LIST:-${ROOT_DIR}/assets/training_data/robotwin.txt}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
OPTIMIZER="${OPTIMIZER:-adamw}"
MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

if [[ -z "${GPU_COUNT}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_COUNT="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
  else
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
fi
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -lt 1 ]]; then
  echo "No CUDA GPUs detected. Set CUDA_VISIBLE_DEVICES or --gpus." >&2
  exit 2
fi

case "${MODE}" in
  lora)
    USE_LORA=true
    # LoRA path matches Robotwin-radeon-cloud: no teachers by default.
    TEACHER_MODE="none"
    DEFAULT_GLOBAL_BATCH="${GPU_COUNT}"
    OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/lora_${GPU_COUNT}gpu_${MAX_STEPS}steps}"
    ;;
  full)
    USE_LORA=false
    DEFAULT_GLOBAL_BATCH="$((GPU_COUNT * MICRO_BATCH_SIZE))"
    OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/full_${TEACHER_MODE}_${GPU_COUNT}gpu_${MAX_STEPS}steps}"
    ;;
  *)
    echo "MODE must be lora or full; got ${MODE}" >&2
    exit 2
    ;;
esac

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${DEFAULT_GLOBAL_BATCH}}"
LOCAL_BATCH_SIZE=$((GPU_COUNT * MICRO_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE % LOCAL_BATCH_SIZE != 0 )); then
  echo "GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by GPU_COUNT * MICRO_BATCH_SIZE (${LOCAL_BATCH_SIZE})" >&2
  exit 2
fi
DEFAULT_GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / LOCAL_BATCH_SIZE))
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_GRAD_ACCUM}}"
if (( MICRO_BATCH_SIZE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS != GLOBAL_BATCH_SIZE )); then
  echo "GLOBAL_BATCH_SIZE must equal MICRO_BATCH_SIZE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS" >&2
  exit 2
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${DATA_LIST}" && ! -d "${DATA_LIST}" ]]; then
  echo "Missing data list/path: ${DATA_LIST}" >&2
  exit 2
fi
if [[ -z "${MODEL_PATH}" || -z "${TOKENIZER_PATH}" ]]; then
  echo "Set MODEL_PATH and TOKENIZER_PATH to local checkpoint directories." >&2
  exit 2
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -e "${TOKENIZER_PATH}" ]]; then
  echo "TOKENIZER_PATH does not exist: ${TOKENIZER_PATH}" >&2
  exit 2
fi

ALIGN_ARGS=()
USE_FUTURE_IMAGE=false
case "${TEACHER_MODE}" in
  none)
    ;;
  full)
    if [[ "${MODE}" != "full" ]]; then
      echo "teacher=full is only supported with --mode full" >&2
      exit 2
    fi
    MOGE_PATH="${MOGE_PATH:-}"
    MORGBD_PATH="${MORGBD_PATH:-}"
    DINO_CKPT="${DINO_CKPT:-}"
    DINO_CONFIG="${DINO_CONFIG:-}"
    for required in "${MOGE_PATH}" "${MORGBD_PATH}" "${DINO_CKPT}" "${DINO_CONFIG}"; do
      if [[ -z "${required}" || ! -e "${required}" ]]; then
        echo "teacher=full requires MOGE_PATH, MORGBD_PATH, DINO_CKPT, DINO_CONFIG" >&2
        exit 2
      fi
    done
    USE_FUTURE_IMAGE=true
    ALIGN_JSON=$(python - <<PY
import json
print(json.dumps({
  "mode": "query",
  "num_task_tokens": 8,
  "depth_loss_weight": 0.004,
  "future_depth_loss_weight": 0.004,
  "use_future_video": True,
  "llm": {"dim_out": 2560, "image_token_size": 8, "image_input_size": 224},
  "depth": {
    "model_type": "MoRGBD",
    "moge_path": "${MOGE_PATH}",
    "morgbd_path": "${MORGBD_PATH}",
    "num_layers": 1,
    "num_heads": 4,
    "dim_head": 32,
    "ff_mult": 1,
    "num_backbone_tokens": 256,
    "token_size": 16,
    "dim_out": 1024,
    "input_size": 224,
    "use_future_depth": True,
    "block_future_depth_to_action": True,
    "detach_future_image_feats": True,
  },
  "video": {
    "ckpt_path": "${DINO_CKPT}",
    "config_path": "${DINO_CONFIG}",
    "attention_mode": "flex_block_causal",
    "input_size": 256,
    "block_suffix_to_future_video": True,
    "block_warmup_steps": 0,
    "block_warmup_gradual": False,
    "share_future_depth_query": True,
    "use_shared_future_task_proj": True,
    "use_current_shared_task_proj": True,
    "shared_query_head_type": "resampler",
    "num_future_frames": 1,
    "use_warmup_frame": True,
    "effective_fps": 1.0,
    "n_blocks": 1,
    "cls_pool": "last",
    "head_type": "resampler",
    "detach_image_feats": True,
    "num_layers": 1,
    "num_heads": 4,
    "dim_head": 32,
    "ff_mult": 1,
    "num_backbone_tokens": 256,
    "dim_out": 1024,
    "target_type": "absolute",
    "future_video_loss_weight": 0.004,
    "use_smooth_l1_loss": False,
    "use_mse_loss": True,
    "mse_loss_weight": 1.0,
    "use_patch_loss": True,
    "use_current_patch_loss": True,
    "use_cosine_loss": False,
    "cosine_loss_weight": 0.2,
    "use_cls_loss": False,
    "cls_loss_type": "mse",
    "cls_loss_weight": 0.2,
    "log_max_samples": 32,
    "log_scale": 16,
  },
  "visual_steps": 5000,
}))
PY
)
    ALIGN_ARGS=(--train.align_params "${ALIGN_JSON}")
    ;;
  *)
    echo "TEACHER_MODE must be none or full; got ${TEACHER_MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "============================================================"
echo " mode=${MODE}  teacher=${TEACHER_MODE}  gpus=${GPU_COUNT}"
echo " micro=${MICRO_BATCH_SIZE}  accum=${GRADIENT_ACCUMULATION_STEPS}  global=${GLOBAL_BATCH_SIZE}"
echo " output=${OUTPUT_DIR}"
echo " model=${MODEL_PATH}"
echo "============================================================"

CMD=(
  torchrun
  --standalone
  --nproc-per-node="${GPU_COUNT}"
  tasks/vla/train_lingbotvla.py
  "${CONFIG}"
  --model.model_path "${MODEL_PATH}"
  --model.config_path "${MODEL_PATH}"
  --model.tokenizer_path "${TOKENIZER_PATH}"
  --model.post_training true
  --data.train_path "${DATA_LIST}"
  --data.use_future_image "${USE_FUTURE_IMAGE}"
  --train.output_dir "${OUTPUT_DIR}"
  --train.optimizer "${OPTIMIZER}"
  --train.use_lora "${USE_LORA}"
  --train.lora_rank "${LORA_RANK}"
  --train.lora_alpha "${LORA_ALPHA}"
  --train.lora_scope action_expert
  --train.train_expert_only false
  --train.data_parallel_mode fsdp2
  --train.data_parallel_replicate_size 1
  --train.data_parallel_shard_size "${GPU_COUNT}"
  --train.micro_batch_size "${MICRO_BATCH_SIZE}"
  --train.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --train.global_batch_size "${GLOBAL_BATCH_SIZE}"
  --train.enable_gradient_checkpointing true
  --train.enable_full_shard true
  --train.enable_fp32 false
  --train.use_compile false
  --train.enable_resume false
  --train.max_steps "${MAX_STEPS}"
  --train.save_steps "${SAVE_STEPS}"
)

if [[ "${#ALIGN_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${ALIGN_ARGS[@]}")
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"
