#!/usr/bin/env bash
# Launch the SMLR SGLang pre-prod inference server (rope-fixed 6-lane) on gpu-host.
#   ./start_sgl_server.sh            # port 8100, GPU3
#   PORT=8100 SGL_GPU=3 ./start_sgl_server.sh
set -euo pipefail
export PATH=~/miniconda3/envs/sglang/bin:$PATH
export CUDA_HOME=/usr/local/cuda
PORT="${PORT:-8100}"
export SGL_GPU="${SGL_GPU:-3}"
export CUDA_VISIBLE_DEVICES="$SGL_GPU"
export CKPT="${CKPT:-/home/user/models/smlr-1b-ml6}"
cd ~/streamingllm
echo "launching SMLR-SGLang server on :$PORT GPU$SGL_GPU model=$CKPT"
exec uvicorn smlr_sgl_server:app --host 0.0.0.0 --port "$PORT" --log-level warning
