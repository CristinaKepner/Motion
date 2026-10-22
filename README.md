# Motion — A3 语义手势动作库：编排、评测、仿真风格化与人工评审

面向 AgiBot A3 的对话手势动作库工具链：

1. **TRABot 式在线编排**：语义函数序列 + 预计时长 → 预算内最长可行前缀 → 五次曲线桥接 → 逐 atom 状态感知执行（论文 14 个公式 + Algorithm 1）。
2. **动作库质量与丰富度评测**：每个片段的自然度、连贯性、可执行性、表现力，带可读的问题标记；库级的语义覆盖、类内多样性、近重复、风格空间覆盖、衔接代价。
3. **来源追溯树**：数据集 → 录制 → atom → 风格变体，逐层汇总质量，能定位“哪条源数据 / 哪种风格在产出不自然、不连贯的动作”。
4. **语义规格核对（名实核对）**：参考 MotionCoach（arXiv:2610.12963），把 17 个语义函数拆成可检查的姿态、接触和先后顺序，核对“贴了这个标签的片段是否真的做到了”，并反查每个片段实际满足哪些语义，作为已验证参考。
5. **仿真风格化生成**：5 种风格预设 + 参考风格迁移 + 左右镜像；生成结果过同一套运动学校验、A3 力矩校验和质量评测，超力矩的自动放慢（控制器约束生成）。
6. **人工评审与裁剪**：单文件评审页（3D 回放、自动问题与规格核对、打分、裁剪入出点、修正标签、规格是否判对），导出 JSON 后由 `apply_reviews.py` 生成新版动作库。

## 目录

```
trabot_a3/
  data.py        A3 38 列 CSV 读写、31 关节约定、中立姿态、限位
  atoms.py       离线建库：式 (4)–(10)，Motion Atom 目录
  bridge.py      式 (13)–(14) 五次曲线桥接
  planner.py     式 (11)–(12) 预算账本、分层 DP + 后向 DP 剪枝 + beam 32
  executor.py    Algorithm 1 状态感知执行、epoch/cancel、neutral recovery
  sim.py         MuJoCo 关节角回放（可选一阶跟踪滞后）、离屏渲染
  quality.py     片段质量评测（SPARC、高频抖动、跳变、起止速度、限位、自碰撞、
                 固定根逆动力学力矩比、SONIC 增益下的 PD 跟踪误差、活动范围、周期性）与库级报告
  semantics.py   17 个语义函数的动作规格（姿态 / 接触 / 时序检查）
  stylize.py     风格预设、参考风格迁移、镜像、生成 + 校验修复
  sourcing.py    来源追溯树与逐层汇总
  paths.py       外部资源路径（环境变量可覆盖）
demo.py               编排演示：4 个回合 → mp4 + CSV + trace.json
build_library.py      评测 → 选基底 → 风格化生成 → 规格核对 → 报告 / 追溯树 / 评审数据包
apply_reviews.py      评审 JSON → 裁剪、改标、重新校验 → out/catalog_v2（含规格校准统计）
benchmark/
  bvh.py / skeleton.py / retarget.py   BVH 解析、各数据集关节名 profile、方向匹配重定向到 A3 17 关节
  metrics.py / runner.py               FKD、L1Div、BC、BAS、语义规格、可执行性、覆盖；数据集/库/对比三种运行方式
  datasets.json / PROTOCOL.md          13 个开源演讲/对话手势数据集目录；评测协议
  tests_roundtrip.py / make_synthetic.py  适配器自检；带音频的合成数据
run_benchmark.py      python run_benchmark.py library | bvh <dir> --profile beat | compare <A> <B>
viz_library.py        官方 A3 模型渲染全部录制与片段，汇总数据来源、数量、仿真计算量、关节覆盖、姿态分布
review_page/
  A3_library_atlas.html   单文件动作库全景（真实 A3 渲染，双击打开）
  A3_motion_review.html   单文件评审台（数据已内嵌，双击打开）
  index.html              页面源码；build_standalone.py 把它和数据打成单文件
out/                       本次运行的结果（目录、评测报告、追溯树、视频、CSV）
```

## 准备

```bash
pip install -r requirements.txt          # mujoco numpy scipy imageio imageio-ffmpeg pillow
mkdir -p external && cd external
git clone https://github.com/AgibotTech/A3-A3U-robot-model           # 官方 A3 MJCF + 网格（非 LFS）
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/AgibotTech/sonic_for_a3   # 只需 a3_data/ 下的 20 条 CSV
cd ..
```

路径也可以用环境变量指定：`A3_MJCF`（a3_t2d5.xml）、`A3_DATA`（CSV 目录）、`A3_OUT`（输出目录）。
渲染视频需要离屏 OpenGL：Linux 上 `apt-get install libosmesa6`（默认 `MUJOCO_GL=osmesa`），有 GPU 时用 `MUJOCO_GL=egl`。

## 运行

```bash
python demo.py --rebuild --no-video     # 建库 + 编排演示（约 5 s）；去掉 --no-video 另出 mp4
python build_library.py                 # 评测 + 风格化生成 + 规格核对（约 2 min）
python review_page/build_standalone.py  # 生成单文件评审台
python viz_library.py && python review_page/build_atlas.py   # 渲染 1,216 帧 A3 并生成动作库全景（约 2 min）
python apply_reviews.py reviews.json    # 评审页导出的 JSON → out/catalog_v2
```

## 本次结果（20 条 A3 录制，`out/library/report.json`）

| | 原始 atom | 仿真生成变体 |
|---|---|---|
| 数量 | 133（丢弃 2 段） | 219（基于 36 个干净 atom） |
| 等级 A / B / C / D | 61 / 4 / 61 / 7 | 177 / 1 / 41 / 0 |
| 通过自身语义标签规格 | **17 / 133** | 47 / 219 |

- **名对、动作不对**：标签由文件名规则提议，32 个“开启话轮”全来自行走摆臂，0 个通过规格。标签必须用动作规格核对，不能只靠名字。
- **主要问题**：几乎不动 61 个、高频抖动 45 个、接近力矩上限 25 个、自碰撞 13 个。全库仅 1% 的片段能在 0.6 s 内从中立姿态接入（录制里手臂多为伸直，中立姿态肘弯约 46°）。
- **风格化**：风格空间覆盖 11 → 12 / 16 格，单侧手片段 2 → 16 个；但语义覆盖仍是 8 / 17，风格化不产生新语义。“指自己”“感谢”在全库没有任何通过规格的片段。
- **力矩约束**：热情 / 活泼 72 个变体中 54 个因 A3 电机力矩超限被自动放慢。
- **规格阈值是第一版**：例如“否定”有 56 个片段通过规格，其中很可能包含行走摆臂这类误判。评审页里的“规格判定是否准确”就是用来校准的，`apply_reviews.py` 会按语义汇总误判通过 / 误判不通过。

## 评审台用法

打开 `review_page/A3_motion_review.html`：

- **质量与丰富度**：结论、名实核对表、语义覆盖、按风格的等级分布、问题类型、风格空间热图、类内多样性。
- **来源追溯树**：可只看问题节点；点 atom 或变体行直接进入评审。
- **人工评审与裁剪**：按自动分从低到高排队；拖动旋转视角；生成变体可叠加原始 atom（灰）对照；用入点 / 出点裁剪；打 4 项分、修正标签、判断规格是否判对、给出结论。

本地打开时评审保存在当前浏览器里，多人评审请各自导出 JSON 后合并给 `apply_reviews.py`（同一片段多人的评分取均值、结论取多数、裁剪点取中位数）。

## 演讲 / 对话手势数据集与评测

`benchmark/datasets.json` 列出 13 个候选（BEAT / BEAT2、Talking With Hands、GENEA 2023、ZEGGS、Trinity、TED 系列、PATS、SHOW、
Seamless Interaction、HiPHI、Bandai Namco、HumanML3D），含规模、格式、标注、许可、下载位置和对 A3 的适配性。
评测把任何来源先转换到 A3 的 17 个手势关节（30 Hz）再打分，协议见 `benchmark/PROTOCOL.md`，结果页 `review_page/A3_gesture_benchmark.html`。

```bash
python run_benchmark.py library                       # A3 现有动作库 + 指标自检
python benchmark/tests_roundtrip.py                   # 适配器自检（合成 BVH、A3 往返）
python run_benchmark.py bvh <folder> --profile beat   # 一个数据集目录：*.bvh（+ 同名 .wav / .tsv，可选 labels.json）
python run_benchmark.py compare <gen_dir> <ref_dir> --profile twh   # 生成结果 vs 真数据（FKD 等）
python benchmark/build_report.py                      # 汇总成单文件报告页
```

profile 可选 `beat` `zeggs` `trinity` `twh`（GENEA 2023）`hiphi` `bandai`；未知骨架按关节名自动匹配。
SMPL-X（BEAT2 / SHOW）和 2D 关键点（TED / PATS）数据还需要各自的适配器。

## 数据来源说明

智元公开的 A3 动作数据只有 `sonic_for_a3/a3_data/agibot_a3` 下的 20 条 CSV（8.1 分钟，58,372 帧 @120 Hz）；
官方说明它们不属于 035 的训练集。035 训练语料未公开，`remote_motions` 的 4 条行走动作为 Git LFS（本环境无法获取），
SONIC 权重托管在 Hugging Face（本环境不可达）。本仓库用到的智元资源是这 20 条 CSV、官方 A3 MJCF 与网格、
SONIC sim2sim 的伺服增益和电机惯量。换成内部动作库时，只需把 `A3_DATA` 指向新的 CSV 目录重跑。

## 已知边界

- 运控闭环用的是固定根逆动力学 + SONIC 伺服增益的 PD 代理，不是 SONIC 策略本身。SONIC 权重在 Hugging Face，需本地下载后用 `sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py --motion out/planned_sequence_30hz.csv --csv-source-fps 30 --csv-frame-stride 1` 接入。
- 数据是 20 条体能 / 步态录制，不是对话手势；补语义需要 BEAT2 重定向或文本条件生成，新数据同样先过规格核对。
- 风格化是可解释的程序化变换，不是学出来的生成模型；它产出的评测分、规格核对和人工评审，可以作为后续生成模型的过滤器和奖励信号。
- 灵巧手尚未挂到 atom 上。
