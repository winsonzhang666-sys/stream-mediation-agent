#!/bin/bash
# 单进程 base + LoRA v4g adapter 共卡 (--enable-lora), TP=4 占后四张卡 4-7。
#   一个 vllm 同时服务:  mediation-raw(基座)  +  mediation-stream-v4g(v4g adapter, 增量diff)
#   8000 = 裸模 mediation-raw 对外裸口 —— 通用对话/工具调用/摘要分类
#   8001 = v4g 业务封装 app_stream_v4g —— 流式调解五块快照
#   拓扑沿用 v4d 那套, 仅把 LoRA 换成 v4g(best checkpoint-200 (eval_loss 0.46254))。
#   工具调用(--enable-auto-tool-choice qwen3_xml)+reasoning 保留(基座裸口用)。
# prompt 一律用 v4g 自己的: app_stream_v4g + conv_prompts_stream_v4g + prompts/ 下的 system
#   (system 由 SYS_FILE 指定, 默认 system_cprime_procfilter_v4.txt; 回滚 SYS_FILE=system_v4g.txt = 训练那份)。
#   全量五块由 apply_diff() 合并; 契约不变: POST /api/mediation/generate {当事人,新对话,当前累积状态?} -> {code,message,data,diff}。
# 拓扑(2026-07-30 变更): GPU 0-3 = ASR(K8s besteffort) 占用; mediation 改用后四张卡 4-7。
#   原"4-7=训练卡, 从不使用"的预留已按运维要求解除。⚠️若训练日后重回 4-7 会与本服务撞卡。
# 回滚: pkill + SYS_FILE 换回上一版 prompt 重启封装层(后端 vLLM 不必重启)。
set -u

KEY="${MEDIATION_API_KEY:?MEDIATION_API_KEY 未设置}"
DIR="${DIR:-$(cd "$(dirname "$0")" && pwd)}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL 未设置(基座路径)}"
ADAPTER="${ADAPTER:-$(cd "$(dirname "$0")/../adapters/v4g" && pwd)}"
LORA_NAME="${LORA_NAME:-mediation-stream-v4g}"
UTIL="${UTIL:-0.90}"

echo "[lora_v4g] $(date) 启动中 util=$UTIL adapter=$ADAPTER name=$LORA_NAME ..."
echo "           8000=裸模 mediation-raw ; 8001=v4g 业务封装 model=$LORA_NAME"

pkill -f 'entrypoints.openai.api_server' 2>/dev/null || true
pkill -f 'uvicorn app:app' 2>/dev/null || true
pkill -f 'uvicorn app_stream' 2>/dev/null || true
sleep 3
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'uvicorn app_stream' 2>/dev/null || true
for i in $(seq 1 15); do
  ps -eo args | grep -E 'vllm.entrypoints.openai.api_server' | grep -v grep >/dev/null || break
  sleep 2
done
sleep 2

# 单进程: 基座 + LoRA v4g, TP=4 占 4,5,6,7, 公网 8000(裸模对外)
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --enable-lora --lora-modules "$LORA_NAME=$ADAPTER" --max-lora-rank 64 --max-loras 1 \
  --tensor-parallel-size 4 --trust-remote-code --dtype bfloat16 \
  --max-model-len 262144 --gpu-memory-utilization "$UTIL" \
  --reasoning-parser qwen3 --gdn-prefill-backend triton \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --served-model-name mediation-raw --api-key "$KEY" \
  --host 0.0.0.0 --port 8000 > /tmp/vllm_lora.log 2>&1 &
echo "[lora_v4g] 后端 vLLM 已拉起 (pid $!) -> 0.0.0.0:8000 (mediation-raw + $LORA_NAME)  log /tmp/vllm_lora.log"

# 业务封装 v4g — 8001; 打到本进程 8000, model=$LORA_NAME; prompt 用 v4g 原生 (app_stream_v4g)
cd "$DIR"
MODEL_NAME="$LORA_NAME" BACKEND_URL="http://127.0.0.1:8000/v1/chat/completions" \
  BACKEND_API_KEY="$KEY" MEDIATION_API_KEY="$KEY" \
  nohup uvicorn app_stream_v4g:app --host 0.0.0.0 --port 8001 > /tmp/mediation_api.log 2>&1 &
echo "[lora_v4g] 封装 v4g 已拉起 (pid $!) -> 0.0.0.0:8001 model=$LORA_NAME  log /tmp/mediation_api.log"

echo "[lora_v4g] 后端加载约 5~6 分钟(权重+图捕获)。封装秒起但要等后端就绪才有响应。"
echo "[lora_v4g] 看日志: tail -f /tmp/vllm_lora.log  ;  封装: /tmp/mediation_api.log"
