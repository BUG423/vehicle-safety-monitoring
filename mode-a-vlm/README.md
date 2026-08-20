# 模式A：大模型（VLM）直接理解车内图片

三条技术路线之一。**定位：发车前一次性安全检查 + 边缘设备上报可疑帧后的云端复核**，
不是 7×24 逐帧实时监控。为什么是这个定位，见 [DESIGN.md](DESIGN.md)。

事件结构完全复用仓库根目录的 [`common/`](../common/) 契约层，`mode=DetectionMode.VLM`，
与模式B/C 的事件可以逐条 diff 对比。

---

## 30 秒跑起来

```bash
cd <仓库根目录>

# 1) 生成合成测试图（1 秒）
python3 mode-a-vlm/scripts/gen_test_images.py --out mode-a-vlm/testdata

# 2) 无需任何 API key，跑通全链路（mock 后端）
python3 mode-a-vlm/scripts/run_pipeline.py mode-a-vlm/testdata/driver_no_seatbelt.jpg \
    --provider mock --scenario driver_no_seatbelt

# 3) 起服务 + 手机页面
python3 -m uvicorn vlm_safety.server:app --app-dir mode-a-vlm --host 0.0.0.0 --port 8000
# 浏览器打开 http://<本机IP>:8000/
```

用真实云端模型（本项目实测用的是硅基流动）：

```bash
export SILICONFLOW_API_KEY=<你的 key>          # 绝不要写进仓库
export SILICONFLOW_VL_MODEL=Qwen/Qwen3-VL-32B-Instruct
python3 mode-a-vlm/scripts/run_pipeline.py mode-a-vlm/evalset/images/12.jpg
```

`VSM_VLM_PROVIDER` 默认是 `auto`：**哪个后端配了凭据就用哪个，全都没有就退回 mock。**
当前用的是哪个后端、是不是模拟输出，会出现在 `/api/health`、每条事件的 `raw_signals`
和手机页面的角标上——不存在「以为在用真模型、其实跑的是 mock」这种情况。

---

## 依赖

```
Python 3.10+
fastapi uvicorn pillow requests        # 必需
opencv-python                          # 可选：人脸脱敏、bench 抽帧
pytest                                 # 可选：跑测试
```

不需要 `python-multipart`——图片走 JSON + base64，手机端 `canvas.toDataURL()` 直接就是这个格式。

---

## 当前实测到什么程度（诚实清单）

| 环节 | 状态 |
|---|---|
| Mock 全链路（解析 → 规则 → 确认 → 双通道告警） | ✅ 跑通，27 条回归测试全绿 |
| **真实云端 VLM 全链路**（硅基流动 Qwen3-VL / GLM-4.5V） | ✅ **跑通**，真实照片上正确检出未系安全带，两条告警通道均返回 `True` |
| FastAPI 服务 + 手机 H5 | ✅ 跑通（mock 与真实后端都验证过） |
| 真实照片验证集（13 张，人工标注） | ✅ 跑通，三个模型横向对比见下 |
| 多帧序列 + PERCLOS 时序疲劳 | ✅ mock 下跑通（4.0 秒确认告警，正常序列零误报） |
| Anthropic / OpenAI / DashScope 后端 | ❌ **未实测**——本环境无这三家的 key。代码按公开接口规范编写，缺 key 时 `health()` 明确报 not ready |
| 本地开源 VLM（`providers/local_hf.py`） | ❌ **未实测通过**。原计划用 A100 跑 Qwen2.5-VL-7B，权重下载到约 6.3GB/16.6GB 时团队决策转向云端 API，下载终止。代码可导入、`health()` 正常，但**从未真实加载过权重、从未产生过一次本地推理** |
| 夜间 / 逆光 / 红外 / 口罩墨镜 鲁棒性 | ❌ 未测 |

---

## 真实照片上的三模型对比

数据集：`evalset/` —— 13 张 Wikimedia Commons 公开授权的**真实**驾驶舱照片，开发者人工标注。
复现：

```bash
python3 mode-a-vlm/scripts/fetch_evalset.py                 # 拉图（图片不入库）
source /root/.config/vsm/env
python3 mode-a-vlm/scripts/eval_models.py \
    --models Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-32B-Instruct zai-org/GLM-4.5V
```

> ⚠️ **13 张、单人标注、无交叉校验。样本量不足以支撑任何精度承诺**，
> 它只回答「真实照片上模型能不能给出可用的结构化判断」和「各模型的保守/激进倾向」。
> 正式评测需要 DMD / StateFarm / AUC Distracted Driver 等真实标注数据集。

| 模型 | 场景判别 | 驾驶员安全带 | 手机使用 | **过度自信** | 平均延迟 | 最慢 | 解析失败 | 单次成本估算 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | 69% | 85% | 85% | **0** | 17.6s | 60.1s | 2/13 | 最低 |
| **Qwen3-VL-32B-Instruct** | **85%** | **85%** | **92%** | 1 | 24.5s | 54.6s | **0/13** | 中 |
| GLM-4.5V | 62% | 69% | 85% | **3** | 21.2s | 60.2s | 2/13 | 中（输出 token 约 2.8 倍） |

平均输入约 1545 token / 输出 344~957 token。

**判定口径**（普通 accuracy 在车载场景下会误导，所以分四档）：

- **正确** = 与人工标注一致，含「人工也判不了 → 模型输出 unknown」
- **过度自信** = 人工都判不了、模型却给了确定结论 → ⚠️ 直接产生误报，**车载场景下最危险**
- **保守** = 人工能判、模型给 unknown → 漏检但不打扰司机，可接受
- **错误** = 模型给出与人工相反的确定结论

### 结论

1. **Qwen3-VL-32B 是当前最优选**：三个轴都最好，且 13 张图**零解析失败**（结构化输出最稳）。
2. **GLM-4.5V 不适合这个场景**：3 次过度自信（人工都看不清安全带，它给了确定结论），
   而且输出 token 是 Qwen 的 2.8 倍（大量思维链），成本与延迟双输。
3. **8B 是"保守但可用"的性价比选项**：过度自信 0 次，代价是场景判别弱一些。
   如果甲方对成本敏感，8B + 提高置信度阈值是可行的折中。
4. **延迟波动比绝对值更值得关注**：同一模型同样大小的图，实测 2.4s ~ 60.2s。
   公有云推理的排队延迟不可控——**这从根本上决定了模式A 不能进入任何有实时性要求的链路**。

### 一个必须记录的教训

第一轮跑出来的结果是「32B 场景判别 15%，还不如 8B 的 77%」。原因不是模型能力，
而是我把 schema 里的枚举写成了 JSON 数组：

```json
"view": ["cabin_front", "cabin_rear", "exterior", ...]     ← 错误示范
```

32B 和 GLM 把它理解成「这个字段的值就是数组」，原样回填 `"view": ["cabin_front"]`，
于是每个字段都被白名单判越界、全部降级 unknown。改成竖线分隔的字符串
`"cabin_front|cabin_rear|..."` 后，32B 从 15% → 85%。

**换模型必须重跑一遍结构化合规性检查，不能默认"更大的模型更听话"。**
详见 [DESIGN.md §2.6](DESIGN.md)。

---

## 接口

服务：`python3 -m uvicorn vlm_safety.server:app --app-dir mode-a-vlm --host 0.0.0.0 --port 8000`

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 手机演示页（响应式，可调摄像头拍照上传） |
| GET | `/api/health` | 当前后端、模型、**是否模拟**、告警配置 |
| GET | `/api/providers` | 可用 VLM 后端列表与当前选择 |
| GET | `/api/scenarios` | mock 后端的预置场景（供演示页切换） |
| POST | `/api/detect` | 单帧检测（`instant` 策略） |
| POST | `/api/detect_sequence` | 多帧序列检测（`temporal` 策略，含 PERCLOS） |
| GET | `/api/alerts` | 最近告警流水（车内 + 后台两条通道） |
| POST | `/api/alerts/clear` | 清空流水（演示用） |
| GET | `/api/cost` | 成本测算（可传 `fleet_size`、`model`） |

`POST /api/detect` 请求体：

```jsonc
{
  "image": "data:image/jpeg;base64,...",   // 也接受裸 base64
  "vehicle_id": "京A12345",                 // 可选
  "scenario": "driver_no_seatbelt",         // 可选，仅 mock 后端生效
  "speed_kmh": 112,                         // 可选，OBD 车速；超速判定只能来自它
  "speed_limit_kmh": 80,
  "engine_on": true,
  "dispatch": true                          // 是否真的发告警
}
```

响应关键字段：

| 字段 | 含义 |
|---|---|
| `events` | `SafetyEvent` 列表（契约层结构，可直接入库） |
| `undecidable` | **判不了的项**，附原因。「看不清」≠「合规」，必须单独呈现 |
| `observations` | 白名单归一化后的结构化视觉观测 |
| `vlm` | 后端、模型、延迟、token、**`simulated`（是否模拟）**、原始回答 |
| `perclos` | 多帧时的闭眼比例统计（时序疲劳的证据量） |
| `alerts` | 每条事件在车内/后台两条通道的投递结果 |
| `notes` | 例如「未提供 VehicleContext：超速类违规本次不参与判定」 |
| `timings_ms` | 预处理 / VLM / 解析 / 规则 / 告警 分段耗时 |

---

## 配置（全部走环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `VSM_VLM_PROVIDER` | `auto` | `auto\|mock\|siliconflow\|anthropic\|openai\|dashscope\|local` |
| `VSM_VLM_MODEL` | 各后端自己的默认值 | 覆盖模型名 |
| `SILICONFLOW_API_KEY` / `SILICONFLOW_BASE_URL` / `SILICONFLOW_VL_MODEL` | — | 硅基流动（本项目实测后端） |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` + `OPENAI_BASE_URL` / `DASHSCOPE_API_KEY` | — | 其余云端后端（未实测） |
| `VSM_LOCAL_MODEL_PATH` / `VSM_LOCAL_DEVICE` | — | 本地 transformers 权重（未实测） |
| `VSM_MAX_IMAGE_EDGE` | `896` | 上传前缩放的最长边，**直接决定 token 成本** |
| `VSM_BLUR_FACES` | `0` | 上云前人脸马赛克（脱敏与疲劳能力互斥，见 DESIGN §7） |
| `VSM_ATTACH_THUMBNAIL` | `1` | 事件是否携带缩略图；关掉则影像留在车端 |
| `VSM_VEHICLE_ID` | `DEMO-VEHICLE-001` | 车辆编号 |
| `VSM_CABIN_MIN_SEVERITY` | `warn` | 车内提醒的最低等级（`info` 会很吵） |
| `VSM_BACKEND_WEBHOOK` | — | 后台上报地址；未配置则打印到 stdout |
| `VSM_ALERT_SPOOL` | `.alert_spool` | 断网落盘目录 |

**API key 只从环境变量读取，仓库里没有任何位置存放 key。**

---

## 目录结构

```
mode-a-vlm/
├── DESIGN.md              技术方案：反幻觉、疲劳时序、能力边界、成本、隐私、路线分工
├── README.md              本文件
├── vlm_safety/
│   ├── config.py          环境变量配置 + provider 自动选择
│   ├── prompts.py         属性抽取式 prompt + 枚举白名单（反幻觉核心）
│   ├── providers/         可插拔后端：mock / siliconflow / anthropic / openai / dashscope / local
│   ├── parser.py          三级容错 JSON 解析 + 白名单归一化
│   ├── rules.py           视觉属性 → 违规的确定性规则 + PERCLOS + OBD 边界
│   ├── pipeline.py        端到端流水线（instant / temporal 两种确认策略）
│   ├── alerting.py        接契约层 AlertDispatcher，车内 + 后台双通道
│   ├── imaging.py         缩放 / 人脸脱敏 / 缩略图
│   ├── cost.py            可执行的成本测算（python3 -m vlm_safety.cost）
│   └── server.py          FastAPI
├── static/index.html      手机页（磨砂玻璃风，真实调用 API）
├── scripts/
│   ├── gen_test_images.py 合成测试图生成
│   ├── run_pipeline.py    命令行端到端
│   ├── fetch_evalset.py   拉取真实照片验证集
│   ├── eval_models.py     多模型横向对比
│   └── run_bench.py       跑 bench/ 公共基准
├── evalset/manifest.json  验证集元数据 + 人工标注（图片不入库）
└── tests/test_mode_a.py   27 条回归测试（走 mock，无需 key）
```

另有单文件离线演示页 [`../mobile-demo/mode-a.html`](../mobile-demo/mode-a.html)，
完全自包含、不连后端、页面上逐环标注哪部分是真实逻辑、哪部分是模拟。

---

## 测试

```bash
python3 -m pytest mode-a-vlm/tests/test_mode_a.py -q     # 27 passed，无需任何 key
```

重点覆盖：模型输出不规范时的解析兜底、**「看不清」绝不被当成「合规」**、
超速只能来自 OBD、插扣欺骗识别、疲劳时序不误报也不漏报、VLM 挂掉时安静失败。

---

## 已知限制

1. 验证集只有 13 张、单人标注 —— 不足以支撑精度承诺。
2. 本地 VLM 与 Anthropic/OpenAI/DashScope 后端**未实测**（见上表）。
3. 未做夜间 / 红外 / 逆光 / 口罩墨镜 的鲁棒性测试。
4. 人脸脱敏用 OpenCV Haar 级联，侧脸漏检明显，生产环境应换专用模型
   （检测器不可用时代码会如实标记 `blur_available=False`，不会假装脱敏成功）。
5. 自一致性投票留了配置项但未启用。
6. 成本数字是按公开标价的量级测算，**不是账单实测**。
7. `bench/` 的合成卡通素材**不能用来评估 VLM 感知精度**（模型会把安全带认成"细长的棍子"）；
   它评的是告警链路与时序逻辑。模式A 的精度证据来自 `evalset/` 的真实照片。
