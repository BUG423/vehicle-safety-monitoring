# 模式B —— 后台服务器实时监测与汇总

车端只做采集与上传，**推理、判定、告警、统计全部在后台**。
一块 GPU 看住整个车队，管理者在一块屏幕上看到全部风险。

技术方案的完整论证见 [`DESIGN.md`](DESIGN.md)；本文只讲**怎么跑、有哪些接口、实测到什么程度、有哪些已知限制**。

---

## 1. 快速开始

```bash
cd mode-b-server

# 1) 下载开源预训练权重（YOLO11 + MediaPipe，公网直接可下，无需登录）
python3 -m modeb.tools.fetch_models

# 2) 一键演示：起后台 + 灌演示数据 + 接入 6 路真实推理的视频源
./demo.sh            # 或 ./demo.sh 12 接入 12 路

# 打开 http://127.0.0.1:8080/          车队看板
#      http://127.0.0.1:8080/cabin?vehicle=京A12345   车机端提醒
```

只想起服务：

```bash
./run.sh                       # 自动选后端：yolo → torchvision → mock
./run.sh --backend yolo        # 强制真模型
MODEB_PORT=9000 ./run.sh       # 换端口
```

### 依赖

环境已预装 torch 2.11+cu128 / torchvision 0.26 / opencv 4.13 / onnxruntime 1.23。
本路线额外需要：

```bash
pip install fastapi "uvicorn[standard]" websockets python-multipart ultralytics mediapipe
```

见 [`requirements.txt`](requirements.txt)。

---

## 2. 怎么把一路车接进来

### 方式一：后台主动去拉（本地文件 / 摄像头 / RTSP）

```bash
curl -X POST http://127.0.0.1:8080/api/v1/sources \
  -H 'Content-Type: application/json' \
  -d '{"vehicle_id":"京A12345","uri":"rtsp://user:pw@10.0.0.7:554/stream",
       "plate":"京A12345","driver_name":"张建国","fps":10}'
```

`uri` 也可以是本地视频文件路径、摄像头设备号（`"0"`），或 `"synthetic"`（合成画面）。

### 方式二：车端主动推（推荐，能穿透 NAT）

```bash
python3 -m modeb.tools.vehicle_agent \
  --server http://127.0.0.1:8080 --vehicle 京A12345 \
  --video ../bench/clips/scenario_a.mp4 --fps 10 --transport ws --with-obd
```

车端 agent **只做采集、JPEG 编码、上传**，不做任何推理。
它同时订阅 `/ws/cabin/<车牌>`，把后台下发的车内提醒打印出来 —— 双向链路一眼可见。

`--transport http` 走逐帧 HTTP POST（穿透性最好），`ws` 走 WebSocket 长连接（省开销）。

---

## 3. 接口一览

### 接入
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/vehicles/register` | 车辆注册（车牌、车队、驾驶员） |
| POST | `/api/v1/vehicles/{id}/heartbeat` | 心跳，body 可带 `speed_kmh` / `speed_limit_kmh` / `gear` 等车身信号 |
| POST | `/api/v1/vehicles/{id}/frame` | HTTP 帧上传，body 为原始 JPEG，元数据放 `X-Meta` 头 |
| WS | `/ws/ingest/{id}` | WebSocket 推帧：二进制消息=JPEG 帧，文本消息=JSON 车身信号 |
| POST | `/api/v1/sources` | 让后台去拉一路 RTSP / 文件 / 摄像头 / 合成源 |
| DELETE | `/api/v1/sources/{id}` | 断开一路 |

### 事件
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/events` | 外部直接上报已确认事件（**模式A/C 混合部署的汇合点**） |
| GET | `/api/v1/events` | 查询，支持 `vehicle_id` / `violation` / `severity` / `review_status` / **`decision`** / `since_s` / `limit` |
| GET | `/api/v1/events/{id}` | 单条详情（含完整 `raw_signals`） |
| GET | `/api/v1/events/{id}/evidence` | 证据帧 JPEG |
| POST | `/api/v1/events/{id}/review` | 复核：`confirmed` / `dismissed` / `appealed` |

### 汇总
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/vehicles` | 车队状态 + 每车当前正在发生的违规 + 流水线健康度 |
| GET | `/api/v1/stats/overview` | 总览 KPI |
| GET | `/api/v1/stats/violations` | 违规类型分布 |
| GET | `/api/v1/stats/vehicles` | 车辆违规排行（按加权扣分，不按条数） |
| GET | `/api/v1/stats/drivers` | 驾驶员安全评分（100 分制，按人聚合） |
| GET | `/api/v1/stats/timeline` | 时段趋势 |
| GET | `/api/v1/stats/data_quality` | **检查完成度**：哪些检查没做完、为什么、哪些车最严重 |
| GET | `/api/v1/system` | 感知后端、降级记录、调度器统计、各车流水线指标 |

### 实时推送
| 路径 | 说明 |
|---|---|
| `/ws/dashboard` | 看板：事件、车辆状态、复核变更 |
| `/ws/cabin/{id}` | 车机：车内提醒（播报文本、重复次数、蜂鸣、横幅颜色） |

### 页面
`/` 车队看板 · `/cabin?vehicle=<车牌>` 车机端 · `/healthz` 健康检查

> **注意**：`/api/v1/events` 返回的是「SafetyEvent + 后台复核字段」。
> 要还原成纯 `SafetyEvent` 请先用 `modeb.server.db.strip_backend_fields()` 剥掉
> `review_status` / `review_note` / `reviewed_at` —— 契约层的 `from_dict()`
> 目前对未知字段是报错而非忽略（已作为修改建议提给汇总方，见 DESIGN.md 第 8 节）。

---

## 4. 「判不了」必须和「没违规」分开

契约层 1.2 引入了 `Decision`，本路线全链路接入：

```
Decision.CONFIRMED     确认违规    → 车内播报 + 后台告警 + 计入统计与评分
Decision.UNDECIDABLE   判不了      → 只上报后台 + 单独统计，不打扰驾驶员、不计入违规
```

**为什么这件事比它看起来重要**：发车前摄像头被贴住，如果后台只是「没收到安全带事件」，
会被读成「检查通过」而放行 —— 这是整套系统最危险的失败模式，而且它是静默的。

本路线在这些情况下产出 `UNDECIDABLE` 而不是「合规」：

| 情况 | 影响的检查 |
|---|---|
| 摄像头被遮挡 / 画面无纹理 | 安全带、疲劳、分心、手机（全部视觉项） |
| 驾驶位未检出人员 | 驾驶员的四项检查 |
| 肩部关键点不可见（侧身/截断） | 安全带 |
| 眼部区域不可见（低头/墨镜/光线不足） | 疲劳 |
| 面部关键点不足，头姿解不出 | 分心 |
| 头部位置圈不定 | 手机 |
| 未接 OBD/GPS 车速信号 | 超速 |

三个关键实现细节：

1. **「判不了」的帧绝不喂进违规判定的滑窗。** 喂 `hit=False` 等于把「看不见」
   当成合规证据，甚至会让一个已经进入违规态的项目被错误判为恢复。
   实现上另起一个确认器专收「判不了」，同样做滑窗投票与最短时长收敛 ——
   偶尔一帧看不清不值得上报，持续看不清才说明检查真的没完成。
2. **统计口径彻底分开。** `overview` / 违规排行 / 驾驶员评分 / 时段趋势
   **只统计 CONFIRMED**；`UNDECIDABLE` 走 `/api/v1/stats/data_quality`，
   作为设备健康与数据质量指标呈现。混在一起会同时犯两个错：误报率虚高，以及更糟的 ——
   漏检被当成合规。
3. **看板能分别回答「这车没问题」和「这车没看清」。** 车队卡片上「未完成」用紫色标签，
   与红/黄的违规标签视觉上就分得开。

实测（全黑画面模拟摄像头被贴住，12 秒）：
产出 1 条 `system.camera_blocked`（CONFIRMED、WARN、播报给驾驶员）
+ 5 条 `UNDECIDABLE`（安全带×2、疲劳、分心、手机，全部 INFO、**不播报**），
且违规判定滑窗内**没有任何样本**（不是「样本全是合规」）。

## 5. 感知后端

| 后端 | 内容 | 适用 |
|---|---|---|
| `yolo` | YOLO11n-pose（人体+17关键点）+ YOLO11n（COCO 手机） | **默认真模型**，吞吐最高 |
| `torchvision` | Keypoint R-CNN R50-FPN + Faster R-CNN MobileNetV3 | 精度参考基线 |
| `cartoon` | 经典 CV 解析合成卡通驾驶舱 | **只对 `bench/` 合成素材有效** |
| `mock` | 脚本化，不读画面 | 冒烟测试与压测灌数据 |
| `auto` | yolo → torchvision → mock 逐级降级，**每次降级都留痕** | 默认 |

叠加可选的 **MediaPipe Face Landmarker**（478 点人脸）后，
疲劳判定从「眼部 ROI 代理」升级为**标准 EAR + PERCLOS**，头姿从 5 点 solvePnP 升级为
模型直接输出的 4×4 变换矩阵。有没有启用会写进每条事件的 `raw_signals.source`，不会含糊。

降级原因可以在 `GET /api/v1/system` 的 `fallbacks` 字段里直接看到。

---

## 6. 实测到什么程度

> **两类实测严格分开，不混用。** 合成素材评的是**告警链路与时序逻辑**，
> 真实素材评的是**真模型的能力**。合成素材上的数字**不代表**真实场景精度。

### 6.1 链路与时序逻辑（`bench/` 公共基准，合成卡通素材，`cartoon` 后端）

```bash
python3 -m modeb.tools.run_clip --video ../bench/clips/scenario_a.mp4 \
    --backend cartoon --fps 15 --out runs/mode_b_events_cartoon.jsonl
cd .. && python3 bench/score.py --truth bench/clips/scenario_a_truth.json \
    --events mode-b-server/runs/mode_b_events_cartoon.jsonl --label 模式B
```

| 指标 | 结果 |
|---|---|
| 检出率 | **4/4（100%）** |
| 告警延迟 | 平均 **5.23 s** / 最慢 **7.73 s** |
| 误报 | **0 次**（0.00 次/分钟） |
| 被眨眼骗到 | **0 次**（素材含 6 次 0.15–0.2 秒眨眼） |
| 被短暂低头骗到 | **0 次**（素材含 3 次 0.5–0.7 秒低头） |

| 违规区间 | 检出 | 延迟 |
|---|---|---|
| driver.no_seatbelt [2–14s] | ✓ | 4.13 s |
| driver.fatigue [20–34s] | ✓ | 7.73 s |
| driver.phone_use [40–48s] | ✓ | 4.93 s |
| passenger.no_seatbelt [6–52s] | ✓ | 4.13 s |

**关于延迟的诚实说明**：告警延迟的大头不是网络，是防误报的确认逻辑
（疲劳规则是 8 秒窗 / 命中率 50% / 最短持续 4 秒）。这部分参数在 `common/` 里，
三条路线完全相同，是主动选择的结果 —— 宁可晚 4 秒报，也不要误报。

**关于干扰项的诚实说明**：6 次眨眼是真实的像素变化（眼部区域真的闭合），
所以「没被眨眼骗到」是有效结论。但 3 次「短暂低头」在素材里是**画了一条线**而不是
真的改变头部姿态，我们的头姿通路在这段素材上等于没有输入，
因此「没被低头骗到」**不构成对头姿分心判定的有效检验**。

### 6.2 真模型能力（真实素材，`yolo` + MediaPipe 后端）

```bash
python3 -m modeb.tools.fetch_samples          # 下载公开真实素材
python3 -m modeb.tools.verify_real_models --backend yolo --short-side 640 --person-thr 0.35
```

**真实照片上的检测**（YOLO11n-pose + YOLO11n + MediaPipe 478 点）：

| 素材 | 人体 | 关键点 | 头部姿态 | MediaPipe |
|---|---|---|---|---|
| 真实正脸照（grace_hopper） | 1 | 7/17 有效 | yaw **-1.0°**（正脸，正确） | EAR **0.283**、闭眼概率 0.075 → 判定「未闭眼」✓ |
| 真实多人照（bus.jpg） | 4 | 最多 17/17 | yaw 6°–11° | EAR 0.174–0.233 |
| 真实多人照（zidane.jpg） | 2 | 11–13/17 | yaw **42.3°**（侧头，正确） | EAR 0.285 |
| 真实手持手机照 | 1 | — | — | **COCO `cell phone` 置信 0.94** ✓ |

**真实视频吞吐**（OpenCV `vtest.avi`，768×576 真实行人，120 帧）：
单帧 **21.4 ms**（P95 22.2 ms）→ **单路 46.7 FPS**。
但平均只检出 0.17 人/帧 —— 这是**远景监控画面**，行人只有几十像素高，
与舱内特写完全不同。**这段素材只用来测吞吐，不用来说明检出能力。**

**真实人像合成的驾驶舱素材**（`tools/make_real_demo_clip.py` 生成，
每一帧都是真实人脸/人体像素，安全带按真模型给出的肩/髋锚点绘制）：

```bash
python3 -m modeb.tools.make_real_demo_clip
python3 -m modeb.tools.run_clip --video samples/real_cockpit.mp4 \
    --backend yolo --face --short-side 640 --fps 15 --out runs/mode_b_events_real.jsonl
cd .. && python3 bench/score.py --truth mode-b-server/samples/real_cockpit_truth.json \
    --events mode-b-server/runs/mode_b_events_real.jsonl --label 模式B真模型
```

| 指标 | 结果 |
|---|---|
| 检出率 | **3/3（100%）** |
| 告警延迟 | 平均 **2.98 s** / 最慢 **4.93 s** |
| 误报 | **0 次** |
| MediaPipe 人脸命中率 | **600/600 帧**（全程有真实 EAR / PERCLOS / 头姿读数） |
| 疲劳/分心误报 | **0 次**（该素材人物睁眼正脸，正确行为就是不告警） |

**这段素材真的部分**：人体检测、17 点关键点、478 点人脸、EAR、头姿、
安全带的斜带对比度读的都是真实图像像素。
**不真的部分**：拍摄条件（光照、视角、镜头畸变、红外补光）与真实舱内摄像头不同；
人是静态照片，没有真实的头部转动与眨眼，因此疲劳/分心在这段素材上只能验证「不误报」，
**无法验证「能检出」**。

### 6.3 单卡并发扩容曲线（实测，A100 80GB）

**纯模型吞吐上限**（摘掉所有流水线，只反复调 `infer_batch`，输入 960×540）：

| 配置 | 批 8 | 批 16 | 10 fps/路时可带 |
|---|---|---|---|
| 短边 480，姿态+物体双模型 | 91.4 fps | **105.5 fps** | 10.6 路 |
| 短边 480，仅姿态模型 | 157.0 fps | **180.3 fps** | 18.0 路 |
| 短边 320，双模型 | 84.5 fps | 114.6 fps | 11.5 路 |
| 短边 640，双模型 | 73.1 fps | 89.2 fps | 8.9 路 |

**完整流水线**（含解码、规则、防误报确认、告警分发、证据帧编码；
每路目标 10 fps，短边 480，最大批 16，4 个后处理线程）：

| 路数 | 总吞吐 | 单路实际 fps | GPU ms/帧 | 平均批 | 端到端 P50 | 端到端 P95 | GPU 利用率 | 显存 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.9 | 9.93 / 10 | 36.2 | 1.0 | **39.6 ms** | 44.1 ms | 49% | 1497 MB |
| 4 | 38.4 | 9.59 / 10 | 23.2 | 3.2 | **89.5 ms** | 265 ms | 62% | 1623 MB |
| 8 | 62.7 | 7.84 / 10 | 15.8 | 16.0 | 512 ms | 592 ms | 57% | 1723 MB |
| 12 | 59.5 | 4.96 / 10 | 17.7 | 16.0 | 546 ms | 733 ms | 61% | 1723 MB |
| 16 | 56.8 | 3.55 / 10 | 18.4 | 16.0 | 581 ms | 854 ms | 61% | 1723 MB |
| 24 | 59.8 | 2.49 / 10 | 17.0 | 16.0 | 587 ms | 681 ms | 60% | 1723 MB |

**推荐运行点 2 fps/路**（带宽成本最优，见 `DESIGN.md` 第 5 节）的实测：

| 路数 | 总吞吐 | 单路实际 fps | 端到端 P50 | 端到端 P95 | 说明 |
|---:|---:|---:|---:|---:|---|
| 8 | 16.2 | **2.03 / 2** | 143 ms | 311 ms | 完全跑满 |
| 16 | 32.4 | **2.03 / 2** | 215 ms | 545 ms | 完全跑满 |
| 24 | 48.4 | **2.02 / 2** | 186 ms | 455 ms | 完全跑满 |
| 32 | 64.3 | **2.01 / 2** | 316 ms | 843 ms | 完全跑满 |
| 40 | 64.9 | 1.62 / 2 | 726 ms | 1781 ms | **开始掉帧** |

「端到端」= 帧生成 → 该帧的判定与告警处理完成，**不含车端编码与网络上行**。

**结论（这是模式B 最需要回答的问题）**

- **10 fps/路 档位：拐点在 4–8 路之间。** 1–4 路每路都能跑满 10 fps、端到端延迟 < 100 ms；
  8 路开始单路掉到 7.8 fps、延迟跳到 500 ms 以上。
  保守取值：**单卡稳定服务约 6 路 @10fps**。
- **2 fps/路 档位：拐点在 32–40 路之间。** 32 路仍然每路跑满 2 fps、P50 316 ms；
  40 路开始掉帧。保守取值：**单卡稳定服务约 32 路 @2fps**。
- 流水线总吞吐在两个档位下都收敛到 **60–65 fps**，说明限制是吞吐而不是路数本身。
  换算关系很简单：**单卡可带路数 ≈ 60 / 每路帧率**。
- 显存完全不是瓶颈：24 路只用 **1.7 GB / 80 GB**。
  换句话说 **A100 80GB 对这个负载是极大的浪费**，同样吞吐用 T4 或 A10 就够，
  单卡成本能降一个数量级。这是选型时应该主动告诉甲方的。

> **一次测量污染的如实记录**：2 fps 档位的第一轮测试中，32 路与 48 路两行出现异常
> （单路掉到 0.32 / 0.06 fps，显存跳到 13.7 GB）。排查发现是**另一位同事的 16 个进程
> 在测试中途占用了同一块 GPU**。上表的 32 / 40 路数据是换到空闲 GPU 后重测的结果，
> 被污染的那两行已作废、不予采用。

**瓶颈诊断（这条比数字本身更有价值）**

推理时 GPU 利用率只有 **49–62%**，且完整流水线（60 fps）明显低于纯模型上限（105 fps）。
说明瓶颈**不在 GPU 算力**，而在单进程内 Python/CPU 侧：
ultralytics 的 letterbox 预处理、NMS、结果对象构造，加上后处理线程与 GPU 线程争抢 GIL。

三条可验证的证据：
1. 去掉第二个（物体）模型，吞吐从 105 → 180 fps（**+71%**）；
   而分辨率从 640 降到 320 只从 89 → 115 fps（**+28%**）。
   **模型调用次数比分辨率更影响吞吐。**
2. 最初把视频解码放在攒批线程里时，16 路以上 GPU 利用率不到 10%；
   改成每路独立解码线程后升到 60%。
3. 后处理平均 26–33 ms/帧，4 个 worker 理论上能撑 130 fps，不是硬瓶颈。

因此正确的优化方向是**减少每帧的 Python 侧推理调用**（把手机检测合并进姿态模型的多任务头）
和**把前后处理搬到 C++ 侧**（TensorRT / Triton）。后者**未实测**，
按经验通常还有 2–3 倍空间，但不能拿经验值当实测数字用。

### 6.4 端到端链路验证（不是「服务起来了」，是断言看板真的收到）

```
[1] 服务已启动
[2] 感知后端: cartoon | 人脸模块: mediapipe_face_landmarker
[3] 看板 WS 已连接
[4] 视频源已接入
[5] 看板收到事件推送 2 条    driver.no_seatbelt/critical、passenger.no_seatbelt/warn，证据帧均有
[6] 车机收到车内提醒 2 条    「请系好安全带」重复3次+蜂鸣、「请提醒乘客系好安全带」重复1次
[7] 告警通道返回值: {'in_cabin': True, 'backend': True}   ← 双通道均断言为 True
[8] 证据帧可下载: 5088 字节
[9] 统计接口 overview / violations / vehicles / drivers / timeline 均返回正确结果
[10] 复核工作流 pending → dismissed
[11] 车队状态: [('E2E-001', online=True, 165 帧)]
[12] 调度器: 159 批 / 165 帧 / 12.47 fps / 后处理 8.37ms
```

用 `python3 -m pytest tests/ -q` 可复现（含 WebSocket 推送断言）。

---

## 7. 已知限制

**必须先说的三条**

1. **断网 = 完全失效。** 推理在后台，网络断了既检测不了也提醒不了当事人。
   事件不会丢（落盘补传），但断网期间车内没有任何提醒。
   隧道/地库/山区占比高的车队，模式B 单独不成立，必须配车端兜底。
2. **安全带判定是启发式，不是模型。** 判别余量在真实素材上很薄（0.114 vs 0.073）。
   能跑通链路、能演示，**不能作为交付给甲方的检测能力**。生产必须训专用模型，
   并与车身安全带扣合开关信号双确认。
3. **超速无法从图像判断。** 必须接 OBD / CAN / GPS。这一点三条路线相同。

**当前不支持的检测项**（都已在代码里显式关闭并注明原因）
- `driver.smoking` —— COCO 无香烟类，姿态代理误报率高到不可用
- `driver.hands_off_wheel` —— 方向盘位置未标定
- `driver.identity_mismatch` —— 事件类型已定义，无实现
- `passenger.child_front_seat` —— 无实现

**未验证的工况**
- 红外/夜间画面（舱内摄像头夜间是红外灰度图，所有阈值都要重新标定）
- 逆光、戴口罩/墨镜、多人遮挡
- 真实舱内摄像头的广角畸变

**未实现的工程能力**
- WebRTC / GB28181 接入（只预留了 `FrameSource` 接口位置）
- 多 GPU 分片、多实例高可用（后台单点故障 = 全车队失去监测）
- TensorRT / Triton 加速
- 证据帧自动过期清理、人脸脱敏、审计日志、分级授权
- 动态帧率升降（可疑时自动升帧率做确认）

**演示数据的标注**
`tools/simulate_fleet.py --mode events` 灌入的事件全部带 `raw_signals.synthetic = true`，
**它们只用于让看板有数据可看，不代表任何检测能力**。
`--mode sources` / `--mode agents` 才是走完整推理链路的真数据。

---

## 8. 代码结构

```
mode-b-server/
├── DESIGN.md              技术方案论证（架构/算法选型/带宽成本/风险/路线分工）
├── run.sh  demo.sh        一键启动 / 一键演示
├── modeb/
│   ├── config.py          全部阈值集中在这里，支持环境变量覆盖
│   ├── compat.py          MediaPipe 在无图形栈容器里的运行时兼容
│   ├── sources/           视频源：拉流 / 推流 / 合成 / 独立解码线程
│   ├── perception/        检测器接口 + YOLO + torchvision + 人脸478点 + 降级实现
│   │   └── analyzers.py   安全带 / 眼睛开合 / 头姿 / 手机 / 遮挡，每项标注可信度
│   ├── engine/            逐帧规则 → 防误报确认 → SafetyEvent → 告警 → 多路调度
│   ├── server/            SQLite + WebSocket + FastAPI + 看板/车机页面
│   └── tools/             模型与素材下载、离线跑片、车端 agent、车队模拟、并发压测
└── tests/                 端到端与单元测试
```

**与 `common/` 的关系**：本路线只负责「感知层」，
确认逻辑（`ViolationConfirmer`）、事件模型（`SafetyEvent`）、
告警通道（`AlertDispatcher`）全部直接复用契约层，没有另造一套。
所有事件的 `mode` 均为 `DetectionMode.SERVER`。

---

## 9. 环境变量

全部可选，默认值见 `modeb/config.py`。

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEB_BACKEND` | `auto` | `yolo` / `torchvision` / `cartoon` / `mock` / `auto` |
| `MODEB_DEVICE` | `cuda:0` | 推理设备 |
| `MODEB_INFER_SHORT_SIDE` | `480` | 推理输入短边，直接决定吞吐与小目标检出 |
| `MODEB_MAX_BATCH` | `8` | 跨车攒批的最大批大小 |
| `MODEB_BATCH_WAIT_MS` | `12` | 攒批等待时间 |
| `MODEB_BELT_THR` | `0.10` | 安全带判定阈值（**换素材必须重新标定**） |
| `MODEB_PERCLOS_THR` | `0.40` | PERCLOS 疲劳阈值 |
| `MODEB_YAW_THR` | `35` | 分心的偏航角阈值（度） |
| `MODEB_DRIVER_SIDE` | `left` | 驾驶位在画面哪一侧 |
| `MODEB_SMOKING_PROXY` | `false` | 抽烟姿态代理（误报高，默认关） |
| `MODEB_PORT` | `8080` | 服务端口 |
| `MODEB_DB` | `runs/modeb.db` | SQLite 路径 |
