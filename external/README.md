# external/

外部资源放这里（已 gitignore）。`trabot_a3/paths.py` 默认在此查找，也可用环境变量覆盖。

```bash
cd external
git clone https://github.com/AgibotTech/A3-A3U-robot-model              # A3_MJCF → a3_t2d5/mjcf/a3_t2d5.xml（含网格，非 LFS）
git clone https://github.com/AgibotTech/sonic_for_a3   # A3_DATA → a3_data/agibot_a3/*.csv
git clone https://github.com/AgibotTech/agibot_x2_urdf
git -C sonic_for_a3 lfs pull
git -C A3-A3U-robot-model lfs pull
git -C agibot_x2_urdf lfs pull
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `A3_MJCF` | `external/A3-A3U-robot-model/a3_t2d5/mjcf/a3_t2d5.xml` | 官方 A3 模型 |
| `A3_DATA` | `external/sonic_for_a3/a3_data/agibot_a3` | A3 动作 CSV 目录（换内部动作库时指向新目录） |
| `A3_OUT` | `results` | 结果目录 |
| `X2_OMNIHAND_URDF` | `external/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf` | OmniHand URDF + meshes，嫁接到 A3 手腕 |

可选：`Bandai-Namco-Research-Motiondataset`（真实动捕手势，CC BY-NC 4.0）、SONIC 035 权重（Hugging Face，接 sim2sim 用）。

## GENEA / SONIC 演示所需资源

035 权重下载自 <https://huggingface.co/sonic-for-a3/sonic/tree/main/035_step200000>，放到 `sonic_for_a3/checkpoints/035_step200000/model_step_200000.pt`。本次 SHA256 为 `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`。实际外部仓库版本与 LFS 清单见 `results/execution_20261027/environment.json`，可据此 checkout 相同提交。

`python scripts/download_genea.py` 从 Zenodo 官方训练 ZIP 确定性等距选择 24 条 main-agent 录制，下载 BVH/WAV/TSV 和 metadata 并检查 ZIP CRC。完整 ZIP 下载、许可证与 manifest 位于 `external/datasets/genea2023/`。音频保留供后续 BC/BAS；本次中文 demo 的音频另由 TTS 生成。

InterHand2.6M 的 5 fps v1 标注备份位于 `external/datasets/interhand26m/annotations/annotations/`，包含 train/val/test 的 3D 关节、相机、MANO、数据索引与骨架说明。RGB 图片未下载；不要把该目录当成完整图像数据集。来源及校验清单见 `results/execution_20261027/interhand_download.json`。

本机 `external/sonic_for_a3` 是指向旁边已验证 checkout 的相对软链；新机器正常 clone 即可。O10 扩展随 Motion 存在 `integrations/a3_counting/`，不依赖上游仓库未提交的本地文件。
