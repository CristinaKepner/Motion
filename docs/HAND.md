# 灵巧手：动作库、标注、流式融合 agent

对应 Notion 方案页第 11 节。代码：`trabot_a3/hand.py`（模型、命令空间、姿态库、风格属性、计时 atom）、`trabot_a3/hand_render.py`（离屏渲染、FK 名实核对）、`trabot_a3/hand_agent.py`（流式融合 agent）、`scripts/hand_demo.py`（建库 + 三个场景 + 视频）。页面：`review_page/dist/A3_hand_library.html`。

## 1. 硬件事实与仿真模型

| 项 | 来源 | 结论 |
|---|---|---|
| A3 真机手 | `sonic_for_a3/.../proto/aimdk/protocol/hal/hand/agi_o10_hand.proto`、`o10_hand_bridge.cpp` | **O10 手**，每侧 10 个位置通道，每通道 0–4096：thumb_rotation / thumb_wiggles / thumb_bent / index_wiggles / index_bent / middle_bent / ring_wiggles / ring_bent / pinky_wiggles / pinky_bent。`a3_runtime_config.yaml` 里 `hand_bridge_enabled: false`，默认未接。 |
| 官方 A3 MJCF | `A3-A3U-robot-model/a3_t2d5/mjcf` | 手腕上只有一只**静态装饰手**（`left/right_hand_Link` mesh，五指张开，无关节）。 |
| 可动手模型 | `agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf` | 智元 OmniHand：每手 16 个旋转关节 + 5 个指尖，带 STL（米制）。拇指 roll/abad/mcp/pip/dip，四指 abad/pip/dip（中指 abad 固定）。 |

`hand.build_a3_hand_mjcf()` 把 OmniHand 子树嫁接到 A3 的 `left/right_wrist_yaw_Link`（URDF 掌坐标 +z 指尖 → A3 +x 前臂方向，拇指侧朝上，掌心朝内），删掉静态装饰手，生成 `results/models/a3_t2d5_omnihand.xml`（nq = 70）。渲染验证：掌心法向 = 掌坐标 +x，手指弯曲朝掌心。

**O10 ↔ OmniHand 映射**（`hand.CHANNEL_JOINTS`）：bent 通道驱动 pip 并按限位比例联动 dip（dip/pip = 1.833/1.571），拇指 bent 驱动 mcp → pip → dip；wiggle/rotation 通道驱动 abad / thumb_roll。归一化 u ∈ [0,1]，`to_o10()` → 0–4096。**极性是假设**（bent 0 = 伸直；wiggle 0.5 = 中立外展），上真机前必须校准，见 `O10_CALIBRATION_NOTE`。

已知运动学限制：OmniHand 拇指的"伸出"方向是掌心法向（对掌方向），不能像人手那样贴掌平面侧向伸出。所以 `thumbs_up` 要靠手腕转到掌心向上才像点赞；`open / five / paper` 的拇指用 roll 0.65 / abad 0.4 压回掌平面。

## 2. 姿态库与标注（高质量标注 schema）

21 个静态姿态（`hand.POSES`）：中式单手数字 1–10、石头/剪刀/布、open/relaxed/fist/point/ok/thumbs_up/pinch/flat。每个姿态带 **可检查的规格**（名实核对，同身体层 `semantics.py` 的思路）：

- `ext`：指尖在掌坐标 z > 0.15 m（拇指：到掌心距离 > 0.11 m）
- `curl`：指尖 z < 0.11 m（拇指：x < 0.03 m，贴掌）
- `pinch`：两指尖距离 < 3.5 cm
- `spread`：两指尖距离 > 3 cm
- `hook`：介于 ext 与 curl 之间且离掌 > 2 cm

`hand_render.check_hand_spec()` 用 MuJoCo FK 做核对。初版姿态有 5 个（七/八/九/OK/平掌）名实不符，被核对抓出并修正；现在 21 × 2 侧全部通过。风格化后再核对一次：**294 个 atom 里 22 个风格化后不通过**（formal/tense/tired 的 spread<0 把捏合推开到 4.3–6.1 cm；tense 的 tension 把 nine 的钩指压成 curl）。这就是"每个属性风格化"要配规格核对的原因。

每个 atom（`HandAtom.meta()`）的标注字段：

```
atom_id, label, label_zh, side, duration
keys[]:        t, phase(prepare/stroke/hold/release), o10[10]       ← 打时间戳的关键帧
annotation:    spec, phases[], contacts, style, style_ops, source, reviewed,
               spec_pass_base, spec_pass_styled, spec_checks{...}, fingertips_palm_m, quality.max_channel_rate_per_s,
               event_id / text / t_arrive / t_target（被 agent 调度后补上）
```

库文件：`results/hand_library/library.json`（294 atom = 21 姿态 × 2 侧 × 7 风格），`poses_sheet.png`（官方 A3 + OmniHand 渲染的 21 个姿态），`style_curves.png`。

## 3. 每一个属性风格化

`hand.HandStyle` 的每个字段都是独立杠杆，预设只是打包：

| 属性 | 作用 | 热情 | 沉稳 | 正式 | 活泼 | 紧张 | 疲惫 |
|---|---|---|---|---|---|---|---|
| tension | bent 通道整体偏移（握紧/放松） | −0.05 | +0.05 | 0 | −0.10 | +0.25 | +0.15 |
| spread | wiggle 通道增益（并拢/张开） | +0.5 | −0.2 | −0.4 | +0.7 | −0.6 | −0.3 |
| tempo | 行程时长倍率 | 1.3 | 0.75 | 0.95 | 1.45 | 1.1 | 0.6 |
| snap | 起手曲线：0 最小 jerk → 1 锐起 + 轻微过冲 | 0.7 | 0 | 0.2 | 0.9 | 0.5 | 0 |
| hold | 保持相加时 | +0.1 s | +0.35 | +0.2 | 0 | 0 | +0.5 |
| anticipation | 先反向预备的幅度比 | 0.15 | 0 | 0 | 0.2 | 0 | 0 |
| ripple_ms | 食指→小指逐指错开（负 = 小指先） | 0 | 40 | 0 | −25 | 0 | 70 |
| thumb_lead_ms | 拇指相对四指提前/滞后 | 0 | 0 | 0 | −40 | 0 | 0 |
| settle | 保持段的微回松 | 0 | 0.2 | 0 | 0 | 0 | 0.35 |

`make_atom()` 生成 prepare → (anticipation) → stroke → hold → release 关键帧，`HandAtom.sample(t)` 按通道应用 ripple / thumb lead 与起手曲线。

## 4. 流式融合 agent（`hand_agent.FusionAgent`）

- 输入：`Event(t_arrive, hand_label, side, t_target, style, hold_s, body_label, text)` 流。`t_target` 是动作要落到的时间戳（重音/拍点），`t_arrive` 是事件到达时间。
- 每只手一条轨道；新事件在当前实时姿态上**抢占**正在执行的 atom（prepare 关键帧从实时姿态起步，不跳变）。身体层由 TRABot 桥接（式 13–14）接入。
- 响应时间模型（每个事件记录）：`latency_ms`（规划耗时，实测 ~0.2 ms）→ `earliest = t_arrive + latency + bridge_min(80 ms)` → `t_start = max(earliest, t_target − stroke)`；来不及时提速（tempo ≤ 1.8×）并记录 `late_ms`。
- `at(t)`：任意时刻的融合状态（body 31 维 + L/R 10 维 + O10 整数 + 当前活动事件），**不用顺序回放**，0.9 ms。`ticks(t_end)` 是 30 Hz 流式生成器（给 `o10_hand_bridge` 消费）。
- `timeline()` 输出全部放置结果、事件日志与响应统计。

演示（`results/hand_demo/`，`demo.mp4`、`contact_sheet.png`、`timeline.json`、`stream_log.jsonl`）：

| 场景 | 内容 | 结果 |
|---|---|---|
| 两手「1+1=2」 | 左手 1 → 右手 1 → 双手平掌（=）→ 双手 2，拍点 0.9 s | 4 个事件、6 次放置，全部准点，响应 350 ms（0.28 s 行程 + 桥接） |
| 石头剪刀布 | 双手同拍三次握拳，第四拍右手出剪刀（活泼）、左手出布（沉稳） | 8 次放置准点，响应 300 ms |
| 数 1–5 | 右手 0.7 s 一拍，热情风格 | 准点 |
| 迟到事件 | OK 手势在目标前 60 ms 才到达 | 提速 1.8×，仍迟到 184 ms，被记录 |
| 回放第五秒 | `agent.at(5.0)` 直接求值 | 0.9 ms；`seek_5s.png / .json` |

## 5. 网络与数据集

本环境只能到 GitHub（git / raw）与 PyPI；Hugging Face、Zenodo、Google Drive、Dropbox、MPI、S3、GCS、figshare、Mendeley 全部不通，Git LFS 也拿不到。手部数据集里只有仓库内直接存文件的能用。可用性见 `docs/DATASETS.md` 的手部小节；真正的手部动捕数据（InterHand2.6M、HanCo、FreiHAND、DexYCB、GRAB、ARCTIC）都在上述被封的站点，要在外网机器下载。

## 6. 下一步

1. 上真机校准 O10 极性与每通道的 0/4096 对应角，把 `hand.CHANNEL_JOINTS` 的比例换成实测。
2. 把 `hand_bridge_enabled` 打开，用 `ticks()` 的 O10 流直接喂 `o10_hand_bridge`；测真实响应时间，替换 `bridge_min`。
3. 手部评审台：把 294 个 atom 接进 `A3_motion_review.html` 的评审流程（现在只有页面展示，没有评审回写）。
4. 用外网机器下载 InterHand2.6M / GRAB，做手部重定向与手部 FKD / 名实核对基准。
5. 手势 + 语音的联合时间戳：把 `t_target` 接到 TTS 的音素时间戳。
