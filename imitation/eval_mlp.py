import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import torch
import numpy as np
from env.unitree_env import UnitreeEnv
from imitation.train_mlp import MLPPolicy


def evaluate(model_path="imitation/mlp_policy.pt", n_episodes=3):
    # 加载归一化参数
    with h5py.File("imitation/go1_data.h5", "r") as f:
        obs_mean = f["obs_mean"][:]
        obs_std = f["obs_std"][:]

    # 加载 MLP
    policy = MLPPolicy()
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

            with torch.no_grad():
                action = policy(
                    torch.tensor(obs_norm, dtype=torch.float32)
                ).numpy()

            action = np.clip(action, -0.5, 0.5)
            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        distance = env.data.qpos[0] - start_x
        print(f"Episode {ep + 1}: 距离 = {distance:.2f} 米")


if __name__ == "__main__":
    evaluate()