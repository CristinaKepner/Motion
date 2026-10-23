# external/

外部资源放这里（已 gitignore）。`trabot_a3/paths.py` 默认在此查找，也可用环境变量覆盖。

```bash
cd external
git clone https://github.com/AgibotTech/A3-A3U-robot-model              # A3_MJCF → a3_t2d5/mjcf/a3_t2d5.xml（含网格，非 LFS）
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/AgibotTech/sonic_for_a3   # A3_DATA → a3_data/agibot_a3/*.csv（只需这 20 条）
git clone https://github.com/AgibotTech/agibot_x2_urdf                     # X2_OMNIHAND_URDF → X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf（灵巧手仿真模型）
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `A3_MJCF` | `external/A3-A3U-robot-model/a3_t2d5/mjcf/a3_t2d5.xml` | 官方 A3 模型 |
| `A3_DATA` | `external/sonic_for_a3/a3_data/agibot_a3` | A3 动作 CSV 目录（换内部动作库时指向新目录） |
| `A3_OUT` | `results` | 结果目录 |
| `X2_OMNIHAND_URDF` | `external/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf` | OmniHand URDF + meshes，嫁接到 A3 手腕 |

可选：`Bandai-Namco-Research-Motiondataset`（真实动捕手势，CC BY-NC 4.0）、SONIC 035 权重（Hugging Face，接 sim2sim 用）。
