# TRABot 公式 → 代码对照

论文《TRABot: Talk, Render, Act》（arXiv:2610.06153v2）及其 40 页解读报告中的 14 个公式与 Algorithm 1，在本仓库的落点。
算例核对（与报告一致）：s(0.25)=0.1035、max s′=1.875、T_b(0.8 rad)=1.5 s、θon/θoff=0.110/0.065、放慢倍率 s=1.224。

| 论文 | 位置 | 说明 |
|---|---|---|
| (1)–(3) 请求 / atom / 入出口姿态 | `trabot_a3/atoms.py: MotionAtom`，`planner.Planner.plan(q, functions, B)` | `q_in / q_out` 取 17 个受控关节（腰 3 + 双臂 14） |
| (4) 活动强度 E_t | `atoms.activity_energy` | 14 个手臂关节，角速度 ÷ 关节活动范围，均方根 |
| 检测副本 25 Hz / 执行副本 30 Hz | `data.A3Motion.resample` + `atoms.detection_copy` | 3 帧中值 + [1,4,6,4,1]/16；切点按时间戳映射到 30 Hz |
| (5)–(7) b, Δ, σ(MAD) | `atoms.background_stats` | Δ < 4σ 整条录制拒绝 |
| (8) 双阈值 | `atoms.background_stats` + `atoms.hysteresis_segments` | κ=3/1.5, α=0.20/0.08, β=0.55/0.25；确认 0.12 s / 0.16 s；缺口 < 0.12 s 合并；hold 规则 |
| (9) 边界代价 | `atoms.boundary_cost / refine_boundaries / split_long_segment` | λ=0.6/0.2/0.2，±0.32 s 搜索，> 0.50 拒绝；> 4 s 用带时长惩罚的 DP 切分 |
| (10) 放慢倍率 s | `atoms.kinematic_report / time_stretch / amplitude_scale` | v≤7、a≤120、j≤2200；限位裕量 0.05 rad；s>3 或 >10 s 丢弃；1.02 裕量仅在需修复时乘 |
| 17 个语义函数 | `atoms.SEMANTIC_FUNCTIONS`，`semantics.SPECS` | 规则提议代替 Qwen3-VL；每个语义有可检查的姿态/接触/时序规格 |
| (11) 预算账本 J | `planner.Planner.total_cost` | 进入桥 + Σd + 过渡桥 + 回收桥 + G(0.35) |
| (12) 最长可行前缀 | `planner.Planner.plan` 分层 DP | `layer_min_cost` 记录每层最低完整成本 |
| beam 32 | `planner.Planner.plan` 后半 | 后向 DP 精确剩余成本剪枝；评分 = 预算利用率 − 重复 − 左右失衡 |
| (13) 五次曲线 | `bridge.quintic_bridge` | s(u)=10u³−15u⁴+6u⁵ |
| (14) 桥接时长 | `bridge.bridge_duration` | T_b ≥ 1.875·max\|Δq\| / 1 rad/s；可选加速度 / jerk 下界 |
| Algorithm 1 | `executor.StateAwareExecutor.run` | 每次只提交一个 atom，桥接从实测姿态出发；结束才 neutral recovery |
| epoch / cancel | `executor.StateAwareExecutor.cancel` | 取消后播完当前 atom 即归位 |
| B = 3·T_hat | `planner.PlannerParams.budget_mult` | |
| BAS 二值 / 高斯 | `benchmark/metrics.bas` | §IV-A，窗 ±0.2 s，σ=0.3 s |
