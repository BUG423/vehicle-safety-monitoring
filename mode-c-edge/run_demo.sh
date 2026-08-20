#!/usr/bin/env bash
# 模式C 一键演示：把「模型下载 → 真模型实测 → 链路评测 → 断网补传」整条跑一遍。
#
#   bash mode-c-edge/run_demo.sh          # 全部
#   bash mode-c-edge/run_demo.sh perf     # 只跑性能实测
#   bash mode-c-edge/run_demo.sh bench    # 只跑公共基准 + 打分
#   bash mode-c-edge/run_demo.sh offline  # 只跑断网落盘补传演示
#   bash mode-c-edge/run_demo.sh real     # 只跑真模型 + 真实素材
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PORT=18080
STEP="${1:-all}"
cd "$HERE"

hr() { printf '\n%s\n' "==============================================================================="; }
say() { hr; echo "▶ $*"; hr; }

stop_receiver() {
  pkill -f "receiver[.]py --port $PORT" 2>/dev/null || true
  sleep 1
}
trap stop_receiver EXIT

# ---------------------------------------------------------------- 依赖
say "步骤 0 / 准备模型与素材（不入库，用时下载）"
python3 tools/fetch_models.py || { echo "模型下载失败，后续 onnx 后端会不可用"; }
if [[ "$STEP" == "all" || "$STEP" == "perf" || "$STEP" == "real" ]]; then
  python3 tools/fetch_assets.py --only-first || echo "真实素材下载失败，perf/real 步骤会跳过"
fi
mkdir -p runtime

# ---------------------------------------------------------------- 性能实测
if [[ "$STEP" == "all" || "$STEP" == "perf" ]]; then
  say "步骤 1 / 算力受限性能实测（真实素材，onnxruntime 限线程）"
  python3 -m edge.bench --frames 60 --threads 1,2,4 --sizes 320x240,640x480 \
      --json-out runtime/bench_perf.json --md-out runtime/bench_perf.txt
fi

# ---------------------------------------------------------------- 真模型 + 真实素材
if [[ "$STEP" == "all" || "$STEP" == "real" ]]; then
  say "步骤 2 / 真模型跑真实人脸视频（YuNet + 106点关键点 + NanoDet）"
  python3 -m edge.main --backend onnx --source assets/head-pose-face-detection-female.mp4 \
      --width 640 --height 480 --fps 12 --threads 1 --max-frames 300 \
      --obd cruise --no-backend --events-out runtime/real_onnx.jsonl
fi

# ---------------------------------------------------------------- 公共基准
if [[ "$STEP" == "all" || "$STEP" == "bench" ]]; then
  say "步骤 3 / 公共基准素材（合成卡通，只评告警链路与时序逻辑，不评感知精度）"
  [[ -f "$ROOT/bench/clips/scenario_a.mp4" ]] || python3 "$ROOT/bench/make_clip.py" --out "$ROOT/bench/clips/scenario_a"
  rm -f runtime/mode_c_events.jsonl
  python3 -m edge.main --backend rule --source "$ROOT/bench/clips/scenario_a.mp4" \
      --width 640 --height 480 --fps 15 --obd cruise --no-backend \
      --events-out runtime/mode_c_events.jsonl
  say "步骤 3b / 按统一 ground truth 打分"
  python3 "$ROOT/bench/score.py" --truth "$ROOT/bench/clips/scenario_a_truth.json" \
      --events runtime/mode_c_events.jsonl --label "模式C 车载嵌入式" \
      --json-out runtime/bench_score.json
fi

# ---------------------------------------------------------------- 断网补传
if [[ "$STEP" == "all" || "$STEP" == "offline" ]]; then
  say "步骤 4 / 断网落盘 + 恢复补传（真的起后台服务，真的断开）"
  [[ -f "$ROOT/bench/clips/scenario_a.mp4" ]] || python3 "$ROOT/bench/make_clip.py" --out "$ROOT/bench/clips/scenario_a"
  rm -rf runtime/spool runtime/offline_demo.jsonl backend/received_events.jsonl
  stop_receiver
  python3 -u backend/receiver.py --port "$PORT" > runtime/receiver.log 2>&1 &
  sleep 2
  echo "后台健康检查: $(curl -s "http://127.0.0.1:$PORT/health")"
  python3 -m edge.main --backend rule --source "$ROOT/bench/clips/scenario_a.mp4" \
      --width 640 --height 480 --fps 15 --obd cruise \
      --events-out runtime/offline_demo.jsonl --offline-from 8 --offline-to 32
  echo
  echo "--- 后台最终收到 ---"
  curl -s "http://127.0.0.1:$PORT/api/stats" | python3 -m json.tool
  echo "--- 本地待补传队列（应为 0）---"
  ls runtime/spool 2>/dev/null | wc -l
  stop_receiver
fi

hr
echo "完成。产物在 mode-c-edge/runtime/ ："
ls -1 runtime 2>/dev/null | sed 's/^/  /'
hr
