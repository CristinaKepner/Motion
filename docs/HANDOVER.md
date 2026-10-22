# 交接文档

更新：2026-10-23。对应 Notion「多模态交互 / A3 语音驱动手势 Demo 与仿真动作库闭环 — 技术方案」第 1–10 节。

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
```

环境：Python 3.11+，`requirements.txt`；离屏渲染需 OSMesa 或 EGL。外部资源见 `external/README.md`；路径由 `trabot_a3/paths.py` 统一管理（`A3_MJCF` / `A3_DATA` / `A3_OUT`）。

## 4. 关键数据与约定

- A3 CSV：120 Hz，38 列，角度制，根位移 cm；关节名与官方 MJCF 一致（`trabot_a3/data.py`）。
- 手势层只控制 17 个关节（腰 3 + 双臂 14）；头和腿保持中立站姿；根固定。中立姿态取官方 MJCF 的 stand keyframe。
- 内部表示：atom / 变体均为 (T, 31) rad @ 30 Hz；导出 CSV 可直接给 `sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py --motion ... --csv-source-fps 30 --csv-frame-stride 1`。
- 可执行性代理：固定根逆动力学（力矩 ÷ MJCF `actuatorfrcrange`）+ 用 SONIC sim2sim 的伺服增益与电机惯量做力矩受限 PD 跟踪。不是 SONIC 策略本身。
- 片段 id：atom = `<录制>__<起>-<止>`；变体 = `<atom>~<风格>`（文件名中 `~` 写作 `__STYLE__`）。

## 5. 主要结论

1. 名对、动作不对：133 个原始 atom 仅 17 个通过自身标签规格。靠文件名/动作名贴标签不可信，入库必须过规格核对。
2. 智元公开的 20 条录制是体能/步态动作，46% 几乎不动、34% 抖动；换源数据是第一杠杆。
3. 真人动捕手势（Bandai）A3 姿态都能摆出来（0.45°），但 27% 力矩超限、35% 自碰撞：真人手势进库必须放慢 + 缩幅，现有风格化修复已覆盖这两步。
4. 风格化能扩风格空间与左右手覆盖，但不产生新语义；缺的 9 类语义（指自己、感谢等）要靠 BEAT 类数据或文本条件生成补。
5. BC 指标在动作节拍密集时饱和，看与随机基线的差距，或用 BAS。

## 6. 已知边界与坑

- SONIC 035 权重在 Hugging Face，本开发环境不可达；运控闭环未真正接入。
- 语义标签为规则提议，未经两轮人工审核；规格阈值需校准（例：Bandai 的挥手 74/97 通过了「否定」规格）。
- 重定向约 40 ms/帧（每帧一个小二乘），小时级数据需先向量化或改解析解。
- 评审台在本地打开时评审只存在当前浏览器；多人共享需发布为组织内页面（带共享存储）。
- 自碰撞检查用官方 MJCF 的碰撞几何，手部为近似；灵巧手未挂到 atom 时间轴。
- `review_page/dist/*.html` 由脚本生成，不要手改；改 `src/` 后重新构建。

## 7. 下一步（按优先级）

1. 在外网机器下载 BEAT 或 GENEA 2023，`scripts/run_benchmark.py bvh <dir> --profile beat` 跑任务 A，按协议门槛筛片段入库。
2. 组织一轮小规模评审（每人 50 段），用「规格判定是否准确」校准 `semantics.py` 阈值；统一中立姿态与录制姿态（当前 99% 片段接入 > 0.6 s）。
3. 本地下载 SONIC 权重，用 sim2sim 替换 PD 代理作为可执行性终检。
4. 写 SMPL-X（BEAT2 / SHOW）与关键点（TED）适配器。
5. 文本 + 风格条件的动作生成模型：用本仓库的评测、规格核对、评审结果作过滤器与奖励。
6. 灵巧手挂到 atom 时间轴。

## 8. 相关链接

- Notion 方案页（多模态交互 / A3 语音驱动手势 Demo 与仿真动作库闭环）
- 论文：TRABot arXiv:2610.06153；MotionCoach arXiv:2610.12963（名实核对思路来源）
- 智元仓库：AgibotTech/sonic_for_a3、AgibotTech/A3-A3U-robot-model、AgibotTech/agibot_x2_urdf
- 评测用真实动捕：BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset（CC BY-NC 4.0，子集标签 `benchmark/bandai_subset_labels.json`）
