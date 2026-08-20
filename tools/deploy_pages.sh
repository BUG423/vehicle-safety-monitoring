#!/usr/bin/env bash
# 把 mobile-demo/ 发布到 gh-pages 分支，供手机通过 GitHub Pages 访问与「添加到主屏幕」。
#
# 源文件保持单文件自包含（便于单独分发），PWA 的 manifest 与 service worker 注册
# 只在部署时注入，不污染源码。
#
#   用法: tools/deploy_pages.sh
#   之后在 GitHub 仓库 Settings → Pages 选择 Branch: gh-pages / root 即可。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SRC="$REPO_ROOT/mobile-demo"
[ -d "$SRC" ] || { echo "找不到 $SRC"; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -r "$SRC"/. "$STAGE"/

# 注入 PWA 能力：manifest 链接 + service worker 注册
python3 - "$STAGE" <<'PYEOF'
import sys, pathlib
stage = pathlib.Path(sys.argv[1])
INJECT = (
    '<link rel="manifest" href="manifest.json">\n'
    '<script>if("serviceWorker" in navigator){'
    'addEventListener("load",function(){navigator.serviceWorker.register("sw.js").catch(function(){})})}'
    '</script>\n'
)
for html in sorted(stage.glob("*.html")):
    s = html.read_text(encoding="utf-8")
    if 'rel="manifest"' in s:
        continue
    # 插在第一个 <style> 之前；没有 <style> 就放到开头（但保持 <title> 仍在最前）
    if "<style>" in s:
        s = s.replace("<style>", INJECT + "<style>", 1)
    else:
        idx = s.find("</title>")
        pos = idx + len("</title>") if idx >= 0 else 0
        s = s[:pos] + "\n" + INJECT + s[pos:]
    html.write_text(s, encoding="utf-8")
    print(f"  注入 PWA: {html.name}")
PYEOF

# 缺失的演示页给出占位，避免 404（各路线尚未合入时）
for m in a b c; do
  f="$STAGE/mode-$m.html"
  [ -f "$f" ] || cat > "$f" <<EOF
<title>模式 $(echo $m | tr a-c A-C) 演示准备中</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#EEF1F6;color:#0B0D12;
font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;text-align:center;padding:24px}
@media(prefers-color-scheme:dark){body{background:#0A0B0E;color:#F2F3F5}}
a{color:#007AFF;text-decoration:none}</style>
<div><p style="font-size:17px">该演示页尚未合入</p>
<p><a href="index.html">‹ 返回</a></p></div>
EOF
done

BRANCH=gh-pages
echo "→ 发布到 $BRANCH 分支"
WT="$(mktemp -d)"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git worktree add --force "$WT" "$BRANCH" >/dev/null
else
  git worktree add --force -b "$BRANCH" "$WT" >/dev/null
  git -C "$WT" rm -rq . 2>/dev/null || true
fi
find "$WT" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -r "$STAGE"/. "$WT"/
touch "$WT/.nojekyll"          # 否则 GitHub Pages 会忽略下划线开头的文件

git -C "$WT" add -A
if git -C "$WT" diff --cached --quiet; then
  echo "  内容无变化，跳过提交"
else
  git -C "$WT" commit -qm "deploy: 发布手机演示页 $(git rev-parse --short HEAD)"
  git -C "$WT" push -u origin "$BRANCH"
  echo "  已推送"
fi
git worktree remove --force "$WT"

echo
echo "完成。下一步（只需做一次）："
echo "  1. 打开 https://github.com/BUG423/vehicle-safety-monitoring/settings/pages"
echo "  2. Source 选 Deploy from a branch，Branch 选 gh-pages / (root)，保存"
echo "  3. 约 1 分钟后访问："
echo "     https://bug423.github.io/vehicle-safety-monitoring/"
echo "     手机 Safari/Chrome 打开后，分享菜单 →「添加到主屏幕」即可像 App 一样使用"
