#!/usr/bin/env bash
# 模式B 后台服务一键启动。
#
#   ./run.sh                     # 自动选后端（yolo → torchvision → mock）
#   ./run.sh --backend cartoon   # 合成素材专用的经典 CV 后端
#   MODEB_PORT=9000 ./run.sh     # 换端口
set -euo pipefail
cd "$(dirname "$0")"

PORT="${MODEB_PORT:-8080}"
BACKEND="${MODEB_BACKEND:-auto}"
DEVICE="${MODEB_DEVICE:-cuda:0}"

# 只用一块 GPU：模式B 的成本论证是「单卡能带多少车」，多卡会把这个数字搅浑
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [ ! -f models/yolo11n-pose.pt ] || [ ! -f models/face_landmarker.task ]; then
  echo "[run] 缺少模型权重，正在下载…"
  python3 -m modeb.tools.fetch_models || echo "[run] 权重下载失败，将按需降级"
fi

echo "[run] 后端=$BACKEND 设备=$DEVICE 端口=$PORT GPU=$CUDA_VISIBLE_DEVICES"
exec python3 -m modeb.server.app --port "$PORT" --backend "$BACKEND" --device "$DEVICE" "$@"
