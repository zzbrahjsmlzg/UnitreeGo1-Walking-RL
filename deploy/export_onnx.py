import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from stable_baselines3 import PPO

# 加载模型
model = PPO.load("models/unitree_final_optimized")
policy = model.policy
policy.eval()  # 关键：设为推理模式

# 定义一个只输出动作均值的包装器
class PolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        # 前向计算，只取动作均值，不采样
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        mean_actions = self.policy.action_net(latent_pi)
        return mean_actions

# 构造包装器
wrapped = PolicyWrapper(policy)
wrapped.eval()

# 假输入
dummy_input = torch.randn(1, 30)

# 导出 ONNX
torch.onnx.export(
    wrapped,
    dummy_input,
    "deploy/unitree_policy.onnx",
    input_names=["obs"],
    output_names=["action"],
    opset_version=18,
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}
)

print("✅ 导出完成：deploy/unitree_policy.onnx")