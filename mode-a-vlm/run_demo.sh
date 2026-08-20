#!/usr/bin/env bash
# 模式A 一键演示入口。
#
#   ./mode-a-vlm/run_demo.sh              起服务（无 key 自动走 mock，全链路照样通）
#   ./mode-a-vlm/run_demo.sh check        只跑一次命令行端到端检测，不起服务
#   ./mode-a-vlm/run_demo.sh test         跑回归测试
#   PORT=9000 ./mode-a-vlm/run_demo.sh    换端口
#
# 有云端 key 时先 `export SILICONFLOW_API_KEY=...`，provider=auto 会自动切到真实模型。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PORT="${PORT:-8000}"
cd "$ROOT"

echo "== 模式A：大模型看图安全检查 =="

# 测试图不入库，每次按需生成（1 秒）
if [ ! -f "$HERE/testdata/driver_no_seatbelt.jpg" ]; then
  echo "[1/3] 生成合成测试图..."
  python3 "$HERE/scripts/gen_test_images.py" --out "$HERE/testdata" >/dev/null
  echo "      -> $HERE/testdata"
else
  echo "[1/3] 测试图已存在，跳过生成"
fi

if [ -n "${SILICONFLOW_API_KEY:-}${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}${DASHSCOPE_API_KEY:-}" ]; then
  echo "[2/3] 检测到云端凭据，将使用真实 VLM 后端"
else
  echo "[2/3] 未检测到任何云端凭据 -> 使用 mock 后端"
  echo "      (VLM 推理是模拟的，其余环节均为真实代码；页面与事件里都会如实标注)"
fi

case "${1:-serve}" in
  check)
    echo "[3/3] 命令行端到端检测..."
    exec python3 "$HERE/scripts/run_pipeline.py" \
        "$HERE/testdata/driver_no_seatbelt.jpg" --scenario driver_no_seatbelt
    ;;
  test)
    echo "[3/3] 回归测试..."
    exec python3 -m pytest "$HERE/tests/test_mode_a.py" -q
    ;;
  serve)
    echo "[3/3] 启动服务 http://0.0.0.0:${PORT}/"
    echo "      手机访问：把地址里的 0.0.0.0 换成本机局域网 IP"
    echo "      注意：手机浏览器调摄像头需要 HTTPS 或 localhost；否则用页面上的「用示例图」"
    exec python3 -m uvicorn vlm_safety.server:app --app-dir "$HERE" \
        --host 0.0.0.0 --port "$PORT"
    ;;
  *)
    echo "未知参数 '$1'，可用：serve（默认） | check | test"; exit 1;;
esac
