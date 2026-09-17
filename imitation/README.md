# 模仿学习对比实验

## 实验目的

对比强化学习（PPO）与模仿学习（BC、Diffusion Policy）在 Unitree Go1 行走任务上的表现差异。

## 实验方法

1. 用已训练好的 PPO 模型（走 7.34 米）采集 200 条成功轨迹
2. 分别训练三种模仿学习模型：
   - 行为克隆（BC）：MLP 直接预测动作
   - 简化版 DP：带噪声预测的 MLP
   - DDPM + Action Chunking：带正确调度器的扩散策略

## 实验结果

| 方法 | 距离 | 说明 |
|------|------|------|
| PPO（专家） | **7.34 米** | 需要环境交互，训练成本高 |
| BC（MLP） | 0.50 米 | 分布偏移，一偏离就崩 |
| 简化 DP | 0.17 米 | 缺少调度器，去噪无效 |
| DDPM + Action Chunking | -0.02 米 | 数据分布太窄，无法泛化 |

## 关键结论

1. **模仿学习在窄分布数据上会失败**
   - PPO 采集的数据只覆盖“成功状态”
   - 模型一旦偏离，无法恢复
   - 这是 IL 在真实机器人上难以落地的根本原因

2. **Action Chunking 不能解决分布偏移**
   - 它只是让动作更平滑
   - 但根本问题在数据分布，不在模型结构

3. **真正的解法需要 DAgger**
   - 让专家在模型失败的状态下补充示范
   - 才能覆盖“偏离 → 纠正”的恢复数据

## 文件说明

| 文件 | 说明 |
|------|------|
| `collect_data.py` | 用 PPO 采集数据 |
| `check_data.py` | 数据检查 |
| `train_mlp.py` | BC 训练 |
| `eval_mlp.py` | BC 评估 |
| `train_dp.py` | 简化版 DP |
| `train_dp_chunk.py` | 带 Action Chunking 的 DP |
| `train_dp_v2.py` | DDPM 版 DP |
| `eval_dp_chunk.py` | 简化版 DP 评估 |
| `eval_dp_v2.py` | DDPM 版 DP 评估 |
| `go1_data.h5` | PPO 采集的原始数据 |
| `go1_data_chunk.h5` | 带 Action Chunking 的数据 |

## 未来改进方向

1. **DAgger**：用专家纠错数据训练
2. **真机数据**：采集人类遥控操作数据
3. **多模态输入**：加入视觉观测