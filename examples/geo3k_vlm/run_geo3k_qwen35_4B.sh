#!/bin/bash

# Qwen3.5-4B VL/RL training on geo3k dataset

# pip install -U transformers

# IMPORTANT: This branch is specially modified for slime's current Megatron
# version and Qwen3.5 from the main Megatron Bridge. Other models are not verified!
# pip install git+https://github.com/coding-famer/Megatron-Bridge-slime.git@qwen35 --no-build-isolation

# Configuration
TRAIN_BACKEND="megatron"

MODEL_NAME="Qwen3.5-4B"

DATASET_NAME=${SLIME_SCRIPT_DATASET_NAME:-"chenhegu/geo3k_imgurl"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")

# ===== Modified: use local base folder, same style as previous script =====
BASE_FOLDER=${BASE_FOLDER:-/gemini/space/gjx/slime_info}
MODEL_PATH=${MODEL_PATH:-"${BASE_FOLDER}/${MODEL_NAME}"}
SAVE_PATH=${SAVE_PATH:-"${BASE_FOLDER}/checkpoint/qwen3.5-4B_slime_geo3k"}
DATASET_PATH=${DATASET_PATH:-"${BASE_FOLDER}/${DATASET_LOCAL_NAME}"}

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# Cleanup
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 slime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 slime
pkill -9 redis

set -ex

export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ===== Use local model and dataset =====
if [ ! -d "${MODEL_PATH}" ]; then
   echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"
   exit 1
fi

if [ ! -d "${DATASET_PATH}" ]; then
   echo "ERROR: DATASET_PATH does not exist: ${DATASET_PATH}"
   exit 1
fi

if [ ! -f "${DATASET_PATH}/train.parquet" ]; then
   echo "ERROR: train data does not exist: ${DATASET_PATH}/train.parquet"
   exit 1
fi

if [ ! -f "${DATASET_PATH}/test.parquet" ]; then
   echo "ERROR: eval data does not exist: ${DATASET_PATH}/test.parquet"
   exit 1
fi

# Common args
CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --load "${MODEL_PATH}"
   --megatron-to-hf-mode bridge
   # --save "${SAVE_PATH}"
   # --save-interval 1
)


ROLLOUT_ARGS=(
   --prompt-data "${DATASET_PATH}/train.parquet"
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 512
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data ${DATASET_LOCAL_NAME} "${DATASET_PATH}/test.parquet"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ===== Modified: 4B does not need 8-GPU EP SGLang setting =====
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
   --sglang-max-running-requests 128
)

# Wandb args (only if WANDB_API_KEY is set)
if [ -n "$WANDB_API_KEY" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime-geo3k-vlm
      --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

MISC_ARGS=(
   --colocate
)

# Backend-specific args
# megatron backend
BACKEND_ARGS=(
   --train-backend megatron

   # ===== Modified: 4B dense model parallel config =====
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   # Packing is not supported for GDN currently
   --qkv-format bshd
   --micro-batch-size 1
)

SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

# ===== Modified: source 4B model config =====
source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Build runtime env
# RUNTIME_ENV_JSON="{
#   \"env_vars\": {
#     \"PYTHONPATH\": \"/root/Megatron-LM/\",
#     \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
#     \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
#   }
# }"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/gemini/space/gjx/slime:/gemini/space/gjx/Megatron-Bridge-slime/src:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SLIME_WAIT_SGLANG_GENERATE_READY\": \"1\",
    \"SLIME_SGLANG_READY_TIMEOUT\": \"900\",
    \"SLIME_SGLANG_READY_INTERVAL\": \"5\"
  }
}"

TENSORBOARD_ARGS=(
   --tensorboard-dir "${SAVE_PATH}/tensorboard"
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}