import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import torch
import numpy as np
from env.unitree_env import UnitreeEnv
from imitation.train_dp_chunk import ChunkDiffusionPolicy


def evaluate(model_path="imitation/dp_chunk_policy.pt", n_episodes=3):
    with h5py.File("imitation/go1_data_chunk.h5", "r") as f:
        obs_mean = f["obs_mean"][:]
        obs_std = f["obs_std"][:]

    policy = ChunkDiffusionPolicy()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    env = UnitreeEnv()
    env.max_steps = 500

    chunk_size = 16

    for ep in range(n_episodes):
        obs, _ = env.reset()
        start_x = env.data.qpos[0]

        step = 0
        while step < 500:
            # 归一化 obs
            obs_norm = (obs - obs_mean) / obs_std
            obs_norm = np.clip(obs_norm, -5.0, 5.0)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)

            # 从纯噪声开始，逐步去噪
            action_flat = torch.randn(1, chunk_size * 12)
            for t in reversed(range(policy.T)):
                with torch.no_grad():
                    noise_pred = policy(obs_t, action_flat, t)
                action_flat = action_flat - noise_pred * (1.0 / policy.T)

            # 解码成 (16, 12) 的动作序列
            action_chunk = action_flat.squeeze(0).numpy().reshape(chunk_size, 12)
            action_chunk = np.clip(action_chunk, -0.5, 0.5)

            # 执行前几步，然后重新规划
            exec_steps = 4
            for k in range(exec_steps):
                action = action_chunk[k]
                obs, reward, terminated, truncated, _ = env.step(action)
                step += 1
                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        distance = env.data.qpos[0] - start_x
        print(f"Episode {ep + 1}: 距离 = {distance:.2f} 米")


if __name__ == "__main__":
    evaluate()