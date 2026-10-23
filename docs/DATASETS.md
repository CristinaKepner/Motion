# 开源演讲 / 对话手势数据集目录

核对日期 2026-10-22。许可与规模以各数据页为准；标 ⚠ 的项目是论文/页面未明确、需要在下载时核对的。

结构化版本：`benchmark/datasets.json`（评测报告页从它生成）。`profile` 是 `scripts/run_benchmark.py --profile` 的取值。

| 数据集 | 类型 | 规模 | 格式 | 标注 | 许可 | 下载位置 | 本环境可达 | profile | 说明 |
|---|---|---|---|---|---|---|---|---|---|
| [BEAT](https://github.com/PantoMatrix/PantoMatrix)<br>ECCV 2022 | 动捕 · 对话/演讲 | 76 h<br>30 人 · 4 种语言 · 8 种情绪 | BVH（120 Hz，含手指）+ ARKit 面部 + 音频 + 文本 | 3200 万帧级标注：情绪、语义相关度；文本对齐 | 非商业研究用 ⚠（论文未注明，以数据页为准） | Hugging Face · Google Drive | blocked | `beat` | 唯一同时带语义相关度与情绪标注的大规模对话动捕；TRABot 论文的源数据。首选。 |
| [BEAT2](https://pantomatrix.github.io/EMAGE/)<br>CVPR 2024 (EMAGE) | 动捕 · 对话/演讲（网格级） | 60 h（Standard 27 h + Additional 33 h）<br>同 BEAT | SMPL-X + FLAME（30 fps）+ 音频 + 文本 | 继承 BEAT；EMAGE 评测协议（FGD / BC / L1Div / MSE / LVD） | 同 BEAT ⚠ | Hugging Face (H-Liu1997/BEAT2) | blocked | `smplx（需 SMPL-X → 关键点适配器）` | 统一到 SMPL-X，便于和生成模型对接；自带可比的基线分数。 |
| [Talking With Hands 16.2M](https://github.com/facebookresearch/TalkingWithHands32M)<br>ICCV 2019 | 动捕 · 双人对话 | 约 20 h 动作（32 场 116 段）；17 场有音频<br>多人，deep / shallow 两类参与者 | BVH（含手指）+ 部分音频 | 对话任务类型；无语义标注 | CC BY-NC 4.0 | GitHub（Git LFS） | blocked | `twh` | 最大的自然对话动捕；GENEA 2022/2023 的基础数据。 |
| [GENEA Challenge 2023](https://zenodo.org/records/8199133)<br>ICMI 2023 | 动捕 · 双人对话（TWH 整理版） | 训练集 8.6 GB + 验证 331 MB + 测试 506 MB<br>TWH 参与者 | BVH（30 fps）+ WAV + 词级 TSV 转写 + metadata.csv | 词级时间对齐转写；说话人 ID；是否含手指 | CC BY-NC 4.0 | Zenodo 8199133 | blocked | `twh` | 直接可用的 BVH + 音频 + 转写三件套，节奏指标（BC/BAS）可以直接算。 |
| [ZEGGS](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS)<br>CGF 2023 (Ubisoft) | 动捕 · 单人独白（风格化） | 134.65 min<br>1 位女演员 · 英语 · 19 种风格 | BVH（60 fps，含手指）+ 音频 | 每段一个风格标签（愤怒、开心、演讲、疲惫…） | 非商业研究 ⚠（Ubisoft 条款） | GitHub（Git LFS） | blocked | `zeggs` | 风格标签最清晰，适合做风格化评测的参考风格来源。 |
| [Trinity Speech-Gesture](https://trinityspeechgesture.scss.tcd.ie/)<br>IVA 2018 / GENEA 2020 | 动捕 · 单人独白 | 约 244 min<br>1 位男演员 · 英语 | BVH（60 fps）+ 音频 | 无 | 签署协议 · 非商业 · 不得再分发 | trinityspeechgesture.scss.tcd.ie | blocked | `trinity` | GENEA 2020 基准数据；许可最严格。 |
| [TED Gesture / TED-Expressive / TED-Culture](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1546765/pdf)<br>ICRA 2019 · CVPR 2022 · Frontiers 2025 | 视频姿态估计 · 演讲 | TED ≈ 97 h 视频；TED-Expressive 同源；TED-Culture 17.5 h · 183 人 · 6 种语言<br>上千位 TED 讲者 | TED：2D/3D 上身 10 点；Expressive/Culture：3D 43 点（13 上身 + 30 手指，ExPose） | 文本转写；TED-Culture 含语言/文化标签 | 研究用；源视频 TED CC BY-NC-ND | Google Drive | blocked | `keypoints（需关键点 → 方向适配器）` | 讲者最多、最像“演讲”；但姿态来自单目估计，抖动和深度误差大，需要先过质量评测。 |
| [PATS](https://github.com/chahuja/pats)<br>ECCV 2020 | 视频姿态估计 · 独白/脱口秀 | 约 251 h<br>25 人 | 2D OpenPose 关键点 + 音频 + 转写 | 文本 | 研究用 ⚠ | Google Drive | blocked | `keypoints2d（无法直接重定向）` | 规模大但只有 2D，不能直接给 A3；可作为文本-手势语义统计来源。 |
| [SHOW (TalkSHOW)](https://talkshow.is.tue.mpg.de/)<br>CVPR 2023 | 视频拟合 SMPL-X · 演讲 | 26.9 h<br>4 人 | SMPL-X（含手指、面部）+ 音频 | 无 | 研究用 · 申请表 | 项目页 | blocked | `smplx` | 全身 SMPL-X，手指与面部一起，适合联调数字脸。 |
| [Seamless Interaction](https://github.com/facebookresearch/seamless_interaction)<br>Meta FAIR | 双人面对面视频 · 对话 | 4,065 h · 64,739 段互动<br>4,284 人 | 音视频 + 转写；身体/脸部参数化表示（发布范围 ⚠） | 16,898 条内在状态/理由标注；人格量表；关系信息 | CC BY-NC 4.0 ⚠ | Hugging Face · GitHub | blocked | `待定（取决于公开的身体表示）` | 规模最大的对话数据，但 3D 身体数据是否随数据发布需要核实。 |
| [HiPHI](https://github.com/noitom-robotics/hiphi)<br>CoRL 2026 (Noitom) | 动捕 · 通用人体动作 + 人-物交互（非演讲） | 617.5 h（含镜像）<br>132 位表演者 | 55 关节 BVH（90 Hz，Y-up，cm）+ 物体轨迹 | 22 个 FrameNet 框架 · 214 个 Frame-LU 动作标签（如 clap、bend、wave） | ModalityNet 开放研究许可（非商业） | Hugging Face | blocked（站点预览可读） | `hiphi` | sonic_for_a3 文档推荐的重定向源；SOMA Retargeter 已有 agibot-a3t3 配置。无语音，但可补“问候/感谢/肯定”等动作的真人参考。 |
| [Bandai Namco Research Motiondataset 1 / 2](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset)<br>arXiv 2306.08861 | 动捕 · 风格化动作（含手势类内容） | ≈ 20 min + ≈ 3.6 h（36,673 + 384,931 帧 @30 fps）<br>3 位专业演员 | BVH（30 fps，21 关节，无手指） | 内容 × 风格标签：挥手、再见、鞠躬、指引、呼唤、回应、举手… × 愤怒/开心/疲惫/老年/自信不足… | CC BY-NC 4.0 | GitHub（直接存于 git，本环境可下载） | ok | `bandai` | 本环境唯一能直接拿到的真实动捕；内容和风格标签正好对应“名实核对”和“风格化”两条评测。 |
| [HumanML3D / Motion-X](https://github.com/EricGuo5513/HumanML3D)<br>CVPR 2022 · NeurIPS 2023 | 文本-动作（通用，非演讲） | HumanML3D 28.6 h · 14,616 段；Motion-X 更大<br>AMASS 等汇总 | SMPL/SMPL-X + 文本描述 | 每段 3 条以上文本描述 | AMASS 各子集许可，需注册 | 注册下载 | blocked | `smplx` | 训练文本条件动作生成模型的标准数据；手势类样本少。 |

适配性评分（0–3：语音同步 / 语义标注 / 手指 / 3D 质量 / 规模）：

| 数据集 | 语音 | 语义 | 手指 | 质量 | 规模 | 合计 |
|---|---|---|---|---|---|---|
| BEAT | 3 | 3 | 3 | 3 | 3 | 15 |
| BEAT2 | 3 | 3 | 3 | 3 | 3 | 15 |
| Talking With Hands 16.2M | 2 | 1 | 3 | 3 | 3 | 12 |
| GENEA Challenge 2023 | 3 | 1 | 2 | 3 | 3 | 12 |
| ZEGGS | 3 | 1 | 3 | 3 | 1 | 11 |
| Seamless Interaction | 3 | 2 | 1 | 2 | 3 | 11 |
| Trinity Speech-Gesture | 3 | 1 | 2 | 3 | 1 | 10 |
| TED Gesture / TED-Expressive / TED-Culture | 3 | 1 | 2 | 1 | 3 | 10 |
| SHOW (TalkSHOW) | 3 | 1 | 2 | 2 | 2 | 10 |
| HiPHI | 0 | 2 | 2 | 3 | 3 | 10 |
| PATS | 3 | 1 | 1 | 1 | 3 | 9 |
| HumanML3D / Motion-X | 0 | 2 | 1 | 2 | 3 | 8 |
| Bandai Namco Research Motiondataset 1 / 2 | 0 | 2 | 0 | 3 | 1 | 6 |

推荐顺序：BEAT / BEAT2 → GENEA 2023 → ZEGGS → TED 系列；HiPHI、Bandai 无语音，可补真人参考姿态。

## 手部（灵巧手）数据集与本环境可达性（2026-10-23 探测）

网络探测：只有 GitHub（`git clone` / `raw.githubusercontent.com`）与 PyPI 可达；Hugging Face、Zenodo、Google Drive、Dropbox、MPI（mpg.de）、S3、GCS、figshare、Mendeley 全部不通，GitHub 的 Git LFS 对象也拿不到。

| 数据集 | 内容 | 格式 | 托管 | 本环境 | 对灵巧手库的用处 |
|---|---|---|---|---|---|
| InterHand2.6M (ECCV 2020) | 2.6M 帧双手交互，3D 关节 | 2D/3D 关键点、MANO | 项目页 / Google Drive | blocked | 手势姿态真值、双手配合（1+1=2 类） |
| FreiHAND / HanCo (ICCV 2019 / 2021) | 单手 RGB + 3D 关节 + MANO | 关键点、MANO | 弗莱堡大学站点 | blocked | 单手姿态分布 |
| GRAB (ECCV 2020) | 全身 + 手 抓取动捕 | SMPL-X / MANO | MPI | blocked | 手指接触与过渡风格 |
| ARCTIC (CVPR 2023) | 双手操作物体动捕 | SMPL-X / MANO | MPI | blocked | 双手协同 |
| DexYCB (CVPR 2021) | 抓取 RGB-D + 手姿态 | 关键点、MANO | Dropbox | blocked | 抓取类手型 |
| BEAT / BEAT2 | 对话动捕（含手指） | BVH / SMPL-X | HF / Drive | blocked | 说话时的手指形态分布 |
| ZEGGS | 风格化独白（含手指） | BVH | GitHub LFS | blocked | 风格对手指的影响 |
| MediaPipe 手部关键点示例仓库 | 少量 21 点 2D/3D 关键点 | CSV / JSON | GitHub（直接存文件） | ok（零散） | 只能做适配器自检 |

结论：灵巧手库现阶段只能靠**程序化姿态 + 仿真 FK 名实核对**（本仓库已做），真实手部数据需在外网机器下载后用 `benchmark/` 的适配器接入。
