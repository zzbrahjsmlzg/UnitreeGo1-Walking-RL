import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import torch
import numpy as np
from env.unitree_env import UnitreeEnv
from imitation.train_dp import SimpleDiffusionPolicy


def evaluate(model_path="imitation/dp_policy.pt", n_episodes=3):
    with h5py.File("imitation/go1_data.h5", "r") as f:
        obs_mean = f["obs_mean"][:]
        obs_std = f["obs_std"][:]

    policy = SimpleDiffusionPolicy()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    env = UnitreeEnv()
    env.max_steps = 500

    for ep in range(n_episodes):
        obs, _ = env.reset()
        start_x = env.data.qpos[0]

        for step in range(500):
            # 归一化 obs
            obs_norm = (obs - obs_mean) / obs_std
            obs_norm = np.clip(obs_norm, -5.0, 5.0)

            obs_t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)

            # 从纯噪声开始，逐步去噪
            action = torch.randn(1, 12)
            for t in reversed(range(policy.T)):
                with torch.no_grad():
                    noise_pred = policy(obs_t, action, t)
                # 简单去噪：减去预测的噪声
                action = action - noise_pred * (1.0 / policy.T)

            action = action.squeeze(0).numpy()
            action = np.clip(action, -0.5, 0.5)

            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        distance = env.data.qpos[0] - start_x
        print(f"Episode {ep + 1}: 距离 = {distance:.2f} 米")


if __name__ == "__main__":
    evaluate()