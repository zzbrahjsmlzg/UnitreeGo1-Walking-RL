import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import onnx
from onnx import numpy_helper
from env.unitree_env import UnitreeEnv

# 从 ONNX 文件加载权重
onnx_model = onnx.load("deploy/unitree_policy.onnx")

# 把权重提取到 dict 里
weights = {}
for initializer in onnx_model.graph.initializer:
    weights[initializer.name] = numpy_helper.to_array(initializer)

# 打印一下，确认权重都拿到了
print("可用权重：")
for k, v in weights.items():
    print(f"  {k}: {v.shape}")


def policy_forward(obs):
    """
    纯 NumPy 前向计算。
    结构：Linear -> Tanh -> Linear -> Tanh -> Linear -> Tanh -> Linear
    """
    x = obs.astype(np.float32)

    # 第 1 层：Linear(30, 256) + Tanh
    x = x @ weights["policy.mlp_extractor.policy_net.0.weight"].T
    x = x + weights["policy.mlp_extractor.policy_net.0.bias"]
    x = np.tanh(x)

    # 第 2 层：Linear(256, 256) + Tanh
    x = x @ weights["policy.mlp_extractor.policy_net.2.weight"].T
    x = x + weights["policy.mlp_extractor.policy_net.2.bias"]
    x = np.tanh(x)

    # 第 3 层：Linear(256, 128) + Tanh
    x = x @ weights["policy.mlp_extractor.policy_net.4.weight"].T
    x = x + weights["policy.mlp_extractor.policy_net.4.bias"]
    x = np.tanh(x)

    # 第 4 层：Linear(128, 12)，不加激活函数
    x = x @ weights["policy.action_net.weight"].T
    x = x + weights["policy.action_net.bias"]

    return x


# 先做一次动作对比，确认纯 NumPy 和 ONNX 输出一致
if __name__ == "__main__":
    # 创建环境
    env = UnitreeEnv()
    env.max_steps = 1000

    obs, _ = env.reset()

    # 用纯 NumPy 推理
    action = policy_forward(obs)
    print(f"\n[NumPy] 第一次动作（前 3 维）: {action[:3]}")

    # 用 ONNX 推理做对比
    import onnxruntime as ort
    session = ort.InferenceSession("deploy/unitree_policy.onnx")
    obs_input = obs.reshape(1, -1).astype(np.float32)
    onnx_action = session.run(["action"], {"obs": obs_input})[0][0]
    print(f"[ONNX]  第一次动作（前 3 维）: {onnx_action[:3]}")

    print(f"差值: {np.abs(action - onnx_action).max()}")

    # 跑一次完整测试
    print("\n开始纯 NumPy 推理...")
    obs, _ = env.reset()
    start_x = env.data.qpos[0]

    for step in range(1000):
    # NumPy 推理
        action_np = policy_forward(obs)
        action_np = np.clip(action_np, -1.0, 1.0)

        # ONNX 推理（用来对比）
        obs_input = obs.reshape(1, -1).astype(np.float32)
        action_onnx = session.run(["action"], {"obs": obs_input})[0][0]
        action_onnx = np.clip(action_onnx, -1.0, 1.0)

        # 打印前 5 步的动作对比
        if step < 5:
            diff = np.abs(action_np - action_onnx).max()
            print(f"步 {step}: 最大差值 = {diff:.2e}")

        # 用 NumPy 动作执行
        obs, reward, terminated, truncated, _ = env.step(action_np)

        if terminated or truncated:
            break

    print(f"\n✅ 纯 NumPy 推理完成，总距离: {env.data.qpos[0] - start_x:.2f} 米")