# Motion — A3 语义手势动作库工具链

面向智元 AgiBot A3 的对话手势动作库：从动作数据到可编排、可评测、可人工评审的动作库，全部在仿真里闭环。

| 能力 | 入口 | 说明 |
|---|---|---|
| 在线编排（TRABot） | `scripts/demo.py` | 语义序列 + 预计时长 → 预算内最长可行前缀 → 五次曲线桥接 → 逐 atom 状态感知执行 |
| 离线建库 | `trabot_a3/atoms.py` | 活动强度、自适应阈值、边界精修、运动学校验与修复（论文式 4–10） |
| 质量评测 + 名实核对 | `trabot_a3/quality.py`, `semantics.py` | 自然度 / 连贯性 / 可执行性 / 表现力；17 个语义的可检查规格 |
| 仿真风格化 | `trabot_a3/stylize.py` | 5 种预设 + 参考风格迁移 + 镜像；力矩超限自动放慢 |
| 来源追溯 | `trabot_a3/sourcing.py` | 录制 → atom → 变体 的逐层质量汇总 |
| 人工评审与裁剪 | `review_page/dist/A3_motion_review.html` | 单文件评审台；`scripts/apply_reviews.py` 回写新版目录 |
| 动作库全景 | `review_page/dist/A3_library_atlas.html` | 官方 A3 模型渲染全部片段、数量、覆盖、姿态分布 |
| 数据集与评测 | `benchmark/`, `scripts/run_benchmark.py` | 13 个开源手势数据集目录；统一到 A3 关节空间的评测套件 |

## 快速开始

```bash
pip install -r requirements.txt
# 外部资源（官方 A3 模型 + 20 条 A3 动作 CSV），见 external/README.md
cd external && git clone https://github.com/AgibotTech/A3-A3U-robot-model && GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/AgibotTech/sonic_for_a3 && cd ..
# Linux 离屏渲染：apt-get install libosmesa6（默认 MUJOCO_GL=osmesa；有 GPU 用 egl）

python scripts/demo.py --rebuild --no-video        # 建库 + 编排演示（约 5 s）
python scripts/build_library.py                    # 评测 + 风格化生成 + 名实核对（约 2 min）
python review_page/build_review.py                 # 评审台 → review_page/dist/
python scripts/viz_library.py && python review_page/build_atlas.py   # 动作库全景（渲染 1,216 帧，约 2 min）
python scripts/run_benchmark.py library && python review_page/build_benchmark.py   # 评测报告
python tests/test_adapters.py                      # 适配器自检
```

## 目录

```
trabot_a3/      核心库：数据约定、建库、桥接、规划、执行、仿真、质量、语义规格、风格化、追溯、路径
benchmark/      数据集 → A3 评测：BVH 解析、关节 profile、重定向、指标、运行器、数据集目录
scripts/        命令行入口（demo / build_library / apply_reviews / viz_library / run_benchmark / make_synthetic_dataset）
review_page/    页面源码 src/、构建脚本、成品 dist/（三个单文件 HTML，数据内嵌，双击打开）
results/        本次运行的结果：catalog/（133 atom）、library/（219 变体 + 评测包）、benchmark/、demo/
tests/          适配器自检（合成 BVH、A3 往返）
docs/           HANDOVER.md（交接）、PROTOCOL.md（评测协议）、FORMULA_MAP.md（公式→代码）、DATASETS.md（数据集目录）
external/       外部资源（gitignore，README 说明获取方式）
```

## 现状一览（20 条智元公开录制）

- 133 个原始 atom，其中只有 17 个通过自身语义标签的规格（标签来自文件名，动作本身是步态/体能动作）；51% 为 C/D 级。
- 219 个仿真风格变体，178 个达到 A/B；风格空间 11 → 12 / 16 格；语义覆盖仍是 8 / 17。
- 真人动捕手势（Bandai Namco 97 段）在 A3 上重定向误差 0.45°，但原速原幅度执行 27% 力矩超限、35% 自碰撞；多样性是智元录制的 3 倍。

详细交接、已知边界与下一步：**`docs/HANDOVER.md`**。
