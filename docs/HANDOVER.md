# 交接文档

更新：2026-10-27。对应 Notion「多模态交互 / A3 语音驱动手势 Demo 与仿真动作库闭环 — 技术方案」第 1–11 节。

## 本轮交付：GENEA + 真实 SONIC

完整执行记录与复现命令见 [`EXECUTION_20261027.md`](EXECUTION_20261027.md)。已合并远端 `c097a50` 的灵巧手库更新，保留原 `FusionAgent` 与手部库；新的 `HandAgent` 用原生 O10 物理模型对词级时间戳做关节跟踪，两套接口没有互相替换。

- 新库：24 条 GENEA 训练录制 → 298 个通过修复门槛的窗口 → **325 atom**；另有 **204 风格变体**，其中 201 为 A/B。旧 133 atom 与旧 352 片段评审页保留。
- 同规则比较：A/B **52.6% → 71.7%**；静止 **45.9% → 0**；规则语义覆盖 **13/17 → 16/17**；语义 × 七种共有风格的空格 **37 → 22 /119**。这些数字不是人工语义准确率。
- 真控制器：10 A + 10 C 均稳定、0 摔倒，动作段上身 RMSE 均值 0.111 / 0.129 rad。不能据此说 C 级更容易摔；代理评分不代替真实运控准入。
- 演示：`results/genea_demo/demo.mp4`，**21 秒、三种身体语义、中文 TTS、O10 1→2→3、SONIC 自由根**；三个词的动作峰时间误差 +33 / −7 / 0 ms，稳定通过。右手 1→2 保持伸展的关节最大偏差 0.00025 rad，没有中途归零。
- 前端：8840 的新入口为 `A3_genea_review.html`；另有 `A3_genea_atlas.html`、`A3_gesture_benchmark.html` 和 `A3_speech_demo.html`。运行 `python scripts/serve_frontend.py --port 8840`。
- Notion 尚未连接；可回填的第 14 节和 13.3 状态放在 `NOTION_SECTION14_20261027.md`，没有把远端更新标成已完成。

## 1. 这个仓库解决什么问题

A3 真机动作库的卡点：数据量不够、训不好、质量差、没法风格化。本仓库把「动作数据 → 可编排的动作库 → 自动评测 → 仿真风格化 → 人工评审 → 回写」做成一条可追溯、可量化的流水线，并给出面向开源演讲/对话手势数据集的统一评测。

## 2. 工作内容与状态

| # | 工作 | 状态 | 代码 | 产物 |
|---|---|---|---|---|
| 1 | TRABot 论文复现：离线建库（式 4–10）、预算规划（式 11–12，DP + 剪枝 beam）、五次曲线桥接（式 13–14）、Algorithm 1 状态感知执行、epoch/cancel | 完成，算例与论文一致 | `trabot_a3/atoms.py bridge.py planner.py executor.py sim.py` | `results/demo/`（trace.json、30 Hz CSV、demo.mp4） |
| 2 | 动作质量评测（自然度 / 连贯性 / 可执行性 / 表现力）+ 库级丰富度 | 完成；阈值为第一版 | `trabot_a3/quality.py` | `results/library/report.json` |
| 3 | 名实核对：17 个语义函数的姿态 / 接触 / 时序规格 | 完成；需用评审数据校准 | `trabot_a3/semantics.py` | 评审包里每片段的 spec 结果 |
| 4 | 仿真风格化：5 预设 + 参考迁移 + 镜像；力矩超限自动放慢 | 完成 | `trabot_a3/stylize.py`, `scripts/build_library.py` | `results/library/*.npz`（219 变体） |
| 5 | 来源追溯树 | 完成 | `trabot_a3/sourcing.py` | `results/library/tree.json` |
| 6 | 人工评审台（3D 回放、裁剪、改标、规格判定）+ 回写脚本 | 完成；本地模式可用，多人共享模式需发布为内部页面 | `review_page/src/review.html`, `scripts/apply_reviews.py` | `review_page/dist/A3_motion_review.html` |
| 7 | 动作库全景（真实 A3 渲染、数量、覆盖、姿态分布） | 完成 | `scripts/viz_library.py`, `review_page/src/atlas.html` | `review_page/dist/A3_library_atlas.html` |
| 8 | 数据集调研（13 个）+ 统一评测套件（适配器、指标、协议、运行器） | 完成；SMPL-X / 2D 关键点适配器未写 | `benchmark/`, `scripts/run_benchmark.py` | `results/benchmark/`, `review_page/dist/A3_gesture_benchmark.html`, `docs/DATASETS.md` |
| 9 | 灵巧手：O10 命令空间 ↔ OmniHand 仿真模型、21 姿态 × 7 风格的手部动作库 + 打时间戳标注 + FK 名实核对、每属性风格化、流式融合 agent（响应时间、抢占、寻址）、三个场景视频 | 完成；O10 极性未上真机校准 | `trabot_a3/hand.py hand_render.py hand_agent.py`, `scripts/hand_demo.py` | `results/hand_library/`, `results/hand_demo/`（demo.mp4）, `review_page/dist/A3_hand_library.html`, `docs/HAND.md` |

## 3. 运行顺序

```
scripts/demo.py --rebuild          → results/catalog/        （建库 + 编排演示）
scripts/build_library.py           → results/library/        （评测、风格化、名实核对、评审包）
review_page/build_review.py        → review_page/dist/A3_motion_review.html
scripts/viz_library.py             → results/viz/（gitignore） → review_page/build_atlas.py → dist/A3_library_atlas.html
scripts/run_benchmark.py library   → results/benchmark/library_report.json
scripts/run_benchmark.py bvh <dir> --profile <name>  → results/benchmark/report_<name>.json
review_page/build_benchmark.py     → review_page/dist/A3_gesture_benchmark.html
scripts/apply_reviews.py reviews.json → results/catalog_v2/
scripts/hand_demo.py                → results/models/a3_t2d5_omnihand.xml, results/hand_library/, results/hand_demo/（渲染约 8 min；--no-video 约 1 min）
review_page/build_hand.py          → review_page/dist/A3_hand_library.html
```

本次环境：Ubuntu 24.04 / Python 3.12 / EGL（无需降级操作系统）；通用环境 Python 3.11+，`requirements.txt`；离屏渲染需 OSMesa 或 EGL。外部资源见 `external/README.md`（灵巧手还需 `AgibotTech/agibot_x2_urdf`）；路径由 `trabot_a3/paths.py` 统一管理（`A3_MJCF` / `A3_DATA` / `A3_OUT`）。

## 4. 关键数据与约定

- 原智元 A3 CSV 为 120 Hz，GENEA 导出为 30 Hz；读入必须显式传入 source_fps。CSV 为 38 列，角度制，根位移 cm；关节名与官方 MJCF 一致（`trabot_a3/data.py`）。
- 离线代理和运动学预览只控制 17 个关节（腰 3 + 双臂 14），其余保持中立且固定根。真实 SONIC 演示是自由根，腿由策略自行控制。中立姿态取官方 MJCF 的 stand keyframe。
- 内部表示：atom / 变体均为 (T, 31) rad @ 30 Hz；导出 CSV 可直接给 `sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py --motion ... --csv-source-fps 30 --csv-frame-stride 1`。
- 可执行性代理：固定根逆动力学（力矩 ÷ MJCF `actuatorfrcrange`）+ 用 SONIC sim2sim 的伺服增益与电机惯量做力矩受限 PD 跟踪。不是 SONIC 策略本身。
- 片段 id：atom = `<录制>__<起>-<止>`；变体 = `<atom>~<风格>`（文件名中 `~` 写作 `__STYLE__`）。

## 5. 历史结论（2026-10-23，当前指标见本轮交付）

1. 名对、动作不对：133 个原始 atom 仅 17 个通过自身标签规格。靠文件名/动作名贴标签不可信，入库必须过规格核对。
2. 智元公开的 20 条录制是体能/步态动作，46% 几乎不动、34% 抖动；换源数据是第一杠杆。
3. 真人动捕手势（Bandai）A3 姿态都能摆出来（0.45°），但 27% 力矩超限、35% 自碰撞：真人手势进库必须放慢 + 缩幅，现有风格化修复已覆盖这两步。
4. 风格化能扩风格空间与左右手覆盖，但不产生新语义；缺的 9 类语义（指自己、感谢等）要靠 BEAT 类数据或文本条件生成补。
5. BC 指标在动作节拍密集时饱和，看与随机基线的差距，或用 BAS。

## 6. 已知边界与坑

- SONIC 035 权重已可用，真实自由根终检已接入。20 段是小规模先导，不是独立随机验证集；本轮未执行真机。
- 语义标签仍未经两轮人工审核。本轮修正否定/问候的时间区间混用，GENEA 问候覆盖降到 0，否定降到 2 个；方向描绘仍有 36.9% 通过率，保留过宽警告。
- 重定向约 40 ms/帧（每帧一个小二乘），小时级数据需先向量化或改解析解。
- 评审台在本地打开时评审只存在当前浏览器；多人共享需发布为组织内页面（带共享存储）。
- 自碰撞检查用官方 MJCF 的碰撞几何，手部为近似。
- 灵巧手：真机 O10 通道极性与 0/4096 对应角是假设，未校准；OmniHand 拇指只能朝掌心法向伸出，点赞需手腕配合；手部库是程序化姿态，无真人手部数据（见 `docs/DATASETS.md` 手部小节）。
- `review_page/dist/*.html` 由脚本生成，不要手改；改 `src/` 后重新构建。

## 7. 下一步（按优先级）

1. 在新 GENEA 评审页组织人工复核，重点复查方向描绘、感谢、指自己，以及自碰撞与抖动片段。
2. 组织一轮小规模评审（每人 50 段），用「规格判定是否准确」校准 `semantics.py` 阈值；统一中立姿态与录制姿态（当前 99% 片段接入 > 0.6 s）。
3. 扩大独立、跨录制的 SONIC 验证集；保留 `merge_controller_metrics` 的真实失稳硬门槛，再用独立标注校准代理。
4. 写 SMPL-X（BEAT2 / SHOW）与关键点（TED）适配器。
5. 文本 + 风格条件的动作生成模型：用本仓库的评测、规格核对、评审结果作过滤器与奖励。
6. 灵巧手：真机校准 O10 → 打开 `hand_bridge_enabled` 用 `FusionAgent.ticks()` 喂 `o10_hand_bridge` → 把 294 个手部 atom 接进评审台（详见 `docs/HAND.md`）。

## 8. 相关链接

- Notion 方案页（多模态交互 / A3 语音驱动手势 Demo 与仿真动作库闭环）
- 论文：TRABot arXiv:2610.06153；MotionCoach arXiv:2610.12963（名实核对思路来源）
- 智元仓库：AgibotTech/sonic_for_a3、AgibotTech/A3-A3U-robot-model、AgibotTech/agibot_x2_urdf
- 评测用真实动捕：BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset（CC BY-NC 4.0，子集标签 `benchmark/bandai_subset_labels.json`）
