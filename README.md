# 车辆安全监测系统 —— 三条技术路线并行验证

对准备开出去的车辆做自动化安全检查：**驾驶员危险驾驶行为**、**乘坐人员安全带**、
以及**对当事人和后台的双向告警**。

甲方需求尚未定型，因此本仓库不押注单一技术路线，而是**并行实现三条路线**并给出可对比的证据，
供后续按甲方实际约束（预算 / 车队规模 / 网络条件 / 隐私要求）做选型。

| 路线 | 目录 | 定位 | 基准告警延迟 | 基准误报 | 断网 |
|---|---|---|---|---|---|
| **A** 大模型理解图片 | [`mode-a-vlm/`](mode-a-vlm/) | 发车前一次性检查、云端复核 | 28 s | 0 | 失效 |
| **B** 后台服务器监测 | [`mode-b-server/`](mode-b-server/) | 车队级集中监管与复核 | **5.23 s** | 0 | 失效 |
| **C** 车载嵌入式设备 | [`mode-c-edge/`](mode-c-edge/) | 断网自治、本地实时告警 | 5.60 s | 0 | **可用** |

三条路线跑同一份素材、同一套标注、同一个打分脚本，因此延迟与误报可横向对比。
**检出率不可比**——素材是合成画面，云端大模型把安全带认成「一根细长的棍子」，
它不在模型的训练分布内。各自在真实素材上的表现见 [`docs/MODE-COMPARISON.md`](docs/MODE-COMPARISON.md)。

## 关键设计：先固化契约，再比较路线

三条路线的差异**只在感知层**。确认逻辑、事件模型、告警协议全部收敛到 [`common/`](common/)：

```
采集 → [感知层：三条路线的唯一差异] → 确认层 → 事件层 → 告警层 → 应用层
              ↑ 可替换                    ↑ 三条路线共用，不重写
```

这样做的直接收益：三条路线的输出可以逐条 diff 对比；换路线只换感知层；
也支持**混合部署**（车载设备初筛 → 可疑帧上传 → 云端大模型复核），这很可能是最终的落地形态。

详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 与 [`common/README.md`](common/README.md)。

## 必须先说清的能力边界

**超速无法从图像判断。** 任何"看图识别超速"的说法都不成立。超速判定必须接入
OBD-II / CAN 总线 / GPS 测速信号，再与电子地图限速比对。三条路线在这点上没有差别，
差别只在于该信号在车端还是云端被消费。仓库中 `vehicle.speeding` 事件的输入是 `VehicleContext`
（车速 / 限速 / 挡位 / 安全带扣合开关），不是图像。

## 防误报是这套系统能否活下来的关键

车载场景中**误报比漏报更致命**——司机被误报打扰几次后就会关掉系统，系统随即归零。
`ViolationConfirmer` 用滑窗投票 + 最短持续时长 + 冷却 + 升级四重机制收敛逐帧噪声。

实测：10% 随机命中率（约等于正常眨眼频率）的噪声输入下，疲劳告警误报 **0 次**；
持续疲劳输入下于 **4.0 秒** 首次告警。

## 告警机制（需求 c）

| 等级 | 车内（提醒当事人） | 后台（告警管理者） |
|---|---|---|
| INFO | 不打扰驾驶员 | 记录 |
| WARN | 语音提醒一次 + 黄色横幅 | 记录 + 统计 |
| CRITICAL | 语音循环 + 蜂鸣 + 红色横幅 | 记录 + 实时推送 |

后台通道内置断网落盘补传——隧道、地库断网是车载常态，事件先落盘、恢复后自动补传。

## 目录结构

```
common/            三条路线共享的契约层（纯标准库，嵌入式端可直接引入）
  schema/          违规类型、安全事件、防误报确认状态机
  alerting/        车内 + 后台双通道告警分发
mode-a-vlm/        模式A：VLM 图片理解
mode-b-server/     模式B：后台服务器实时监测汇总
mode-c-edge/       模式C：车载嵌入式设备
mobile-demo/       手机演示页（单文件自包含，可直接在手机浏览器打开）
docs/              总体架构、三路线对比与选型建议
```

## 手机演示

**https://bug423.github.io/vehicle-safety-monitoring/**

手机浏览器打开后，分享菜单 →「添加到主屏幕」，即可像 App 一样使用（PWA，离线可用）。
各演示页都标注了哪部分是真实逻辑、哪部分是模拟——请以页面上的标注为准，不要把演示当作实测。

页面源码在 [`mobile-demo/`](mobile-demo/)，单文件自包含、无外部依赖；
`tools/deploy_pages.sh` 负责发布到 `gh-pages` 分支。

## 本地运行

```bash
# 契约层与全仓测试（89 项）
python3 -m pytest tests/ mode-a-vlm/tests mode-b-server mode-c-edge -q

# 模式 A：无 key 走 mock，配了 SILICONFLOW_API_KEY 则调真实云端模型
./mode-a-vlm/run_demo.sh check

# 模式 B：后台服务 + 车队看板
bash mode-b-server/run.sh

# 模式 C：车载端全链路（含性能实测与断网补传）
bash mode-c-edge/run_demo.sh

# 公共评测基准：生成带标注素材并给任一路线打分
python3 bench/make_clip.py --out bench/clips/scenario_a
python3 bench/score.py --truth bench/clips/scenario_a_truth.json --events <事件>.jsonl --label 模式C
```

各路线的详细运行方式、接口说明与已知限制见各自的 `README.md` 与 `DESIGN.md`。

## 凭据

云端模型的 API key 通过环境变量注入，**绝不入库**：

```bash
export SILICONFLOW_API_KEY=<你的 key>
export SILICONFLOW_VL_MODEL=Qwen/Qwen3-VL-32B-Instruct
```

`.gitignore` 已覆盖 `.env`/`*.env`/`*apikey*` 等形式。提交前可自查：
`git diff --cached | grep -i 'sk-'`。
