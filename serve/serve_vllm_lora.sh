#!/bin/bash
# 流式调解 diff 模型 serving —— 不 merge,vLLM 直接加载 base + LoRA adapter(v4g)。
#
# nothink 模型:客户端每次请求**必须**带 chat_template_kwargs{"enable_thinking": false}
#   (见 serve/demo_infer.py;这是 per-request 客户端参数,不是 serve 端 flag)。
#
# 前置:
#   - 自备基座 Qwen3.6-35B-A3B(MoE / qwen3_5_moe),并让 transformers/vLLM 支持该架构;
#   - system prompt 必须用 prompts/system_v4g.txt(train==serve 单一事实源)。
#
# 用法:
#   BASE_MODEL=/path/to/Qwen3.6-35B-A3B TP=4 bash serve/serve_vllm_lora.sh
#   起来后:  python serve/demo_infer.py
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen3.6-35B-A3B}"        # 基座路径 / 名称(需自备)
ADAPTER_PATH="${ADAPTER_PATH:-./adapters/v4g}"     # 本仓适配器目录
SERVED_NAME="${SERVED_NAME:-mediation-stream-v4g}" # OpenAI API 里的 model 名
TP="${TP:-4}"                                      # tensor 并行度(按你的卡数改)
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --enable-lora \
  --lora-modules "${SERVED_NAME}=${ADAPTER_PATH}" \
  --max-lora-rank 64 \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --host "$HOST" --port "$PORT"

# 说明:
#   --reasoning-parser qwen3  与 nothink 客户端参数配合;
#   后端相关 flag(如 --gdn-prefill-backend 等)按各自 infra 需要自行追加,与本模型无关。
