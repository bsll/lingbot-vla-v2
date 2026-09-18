# CUDA LoRA / Full-SFT (RoboTwin)

Branch helper for NVIDIA CUDA training. LoRA wiring is adapted from
`Robotwin-radeon-cloud` (`lingbot-vla-v2-rocm.patch`), with HIP/ROCm bits removed.

## What changed

- `tasks/vla/train_lingbotvla.py`: `--train.use_lora` and PEFT injection on action-expert attention
- `lingbotvla/utils/lora_utils.py`: regex target modules + return model
- `scripts/train_robotwin_cuda.sh`: one launcher for `lora` and `full`
- `scripts/merge_lora_dcp.py`: merge LoRA DCP → HF weights
- `configs/vla/robotwin/cuda_train.yaml`: shared CUDA-friendly defaults

## LoRA recipe (same idea as radeon-cloud)

- Freeze base weights, train LoRA on `qwen_expert` Q/K/V/O only
- Default: **no** depth/video teachers (`align_params: {}`)
- `micro_batch_size=1`, AdamW, FSDP2 full shard, gradient checkpointing

## Quick start

```bash
# LoRA (recommended first on 1x 48G/80G)
export MODEL_PATH=/path/to/lingbot-vla-v2-6b
export TOKENIZER_PATH=/path/to/Qwen3-VL-4B-Instruct
export DATA_LIST=assets/training_data/robotwin.txt
export CUDA_VISIBLE_DEVICES=0

bash scripts/train_robotwin_cuda.sh --mode lora --max-steps 100

# Merge for deploy
python scripts/merge_lora_dcp.py \
  --checkpoint output/lora_1gpu_100steps/checkpoints/global_step_100 \
  --training-output output/lora_1gpu_100steps \
  --base-model "$MODEL_PATH" \
  --output output/lora_1gpu_100steps/merged_hf_ckpt \
  --rank 8 --alpha 16
```

```bash
# Full-parameter without teachers
bash scripts/train_robotwin_cuda.sh --mode full --teacher none --gpus 1 --max-steps 100

# Full-parameter with teachers (multi-GPU recommended)
export MOGE_PATH=/path/to/moge2-vitb-normal.pt
export MORGBD_PATH=/path/to/morgbd.pt
export DINO_CKPT=/path/to/teacher_step_10000.pth
export DINO_CONFIG=/path/to/dino_video/config.yaml
bash scripts/train_robotwin_cuda.sh --mode full --teacher full --gpus 4 --max-steps 1000
```

Batch rule:

```text
global_batch_size = micro_batch_size × num_gpus × gradient_accumulation_steps
```

## Risks / caveats

See [`lora_readme.md`](./lora_readme.md) for known footguns (merge rank/alpha, frozen action heads, `all_attention` VRAM, PEFT HF export, `post_training`, data placeholders, etc.).
