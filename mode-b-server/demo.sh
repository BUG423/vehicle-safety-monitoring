#!/usr/bin/env bash
# 一键演示：起后台 → 灌历史数据 → 接入真实推理的视频源 → 打开看板。
#
#   ./demo.sh                # 默认 6 路合成源 + 240 条历史事件
#   ./demo.sh 12             # 12 路
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-6}"
PORT="${MODEB_PORT:-8080}"
BACKEND="${MODEB_BACKEND:-cartoon}"
CLIP="../bench/clips/scenario_a.mp4"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p runs
if [ ! -f "$CLIP" ]; then
  echo "[demo] 生成 bench 素材…"
  (cd .. && python3 bench/make_clip.py --out bench/clips/scenario_a)
fi

echo "[demo] 启动后台（backend=$BACKEND, port=$PORT）…"
python3 -m modeb.server.app --port "$PORT" --backend "$BACKEND" > runs/server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for i in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null || { echo "[demo] 后台没起来，看 runs/server.log"; exit 1; }

echo "[demo] 灌入模拟历史事件（仅用于看板演示，带 synthetic 标记）…"
python3 -m modeb.tools.simulate_fleet --server "http://127.0.0.1:$PORT" \
        --mode events --n 12 --events 240 >/dev/null

echo "[demo] 接入 $N 路真实推理的视频源（走完整链路）…"
python3 -m modeb.tools.simulate_fleet --server "http://127.0.0.1:$PORT" \
        --mode sources --n "$N" --video "$CLIP" --fps 8

echo
echo "===================================================================="
echo "  车队看板   http://127.0.0.1:$PORT/"
echo "  车机端     http://127.0.0.1:$PORT/cabin?vehicle=<车牌>"
echo "  系统状态   http://127.0.0.1:$PORT/api/v1/system"
echo "  Ctrl-C 结束"
echo "===================================================================="
wait $SRV
