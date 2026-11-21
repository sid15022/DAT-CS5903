#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-pilot}
if (($#)); then shift; fi
case "$mode" in
    pilot|suite|overhead) ;;
    *) echo "Usage: ./run_gpu.sh {pilot|suite|overhead} [options]" >&2; exit 2 ;;
esac
workers=${WORKERS:-1}
backend=${DAT_BACKEND:-vllm-http}
results=${DAT_RESULTS:-"$PWD/gpu-runs"}
cache=${DAT_MODEL_CACHE:-"$HOME/.cache/huggingface"}
scratch=${DAT_KV_SPOOL:-"$results/spool"}
mkdir -p "$results" "$cache" "$scratch"
results=$(realpath "$results")
cache=$(realpath "$cache")
scratch=$(realpath "$scratch")
image=${DAT_IMAGE:-dat-failslow:http-v0.10.2}
if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker build -f Dockerfile.gpu -t "$image" .
fi
args=(--rm --gpus all --shm-size 4g --hostname dat-pd
    -v "$results:/results" -v "$scratch:/kv" -v "$cache:/root/.cache/huggingface")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then args+=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"); fi
exec docker run "${args[@]}" "$image" "$mode" --backend "$backend" --workers "$workers" \
    --out "/results/${mode}-${workers}x${workers}" --spool /kv "$@"
