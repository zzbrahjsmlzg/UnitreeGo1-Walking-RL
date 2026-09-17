import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import h5py
from env.unitree_env import UnitreeEnv
from stable_baselines3 import PPO


def collect(n_episodes=200, chunk_size=16, save_path="imitation/go1_data_chunk.h5"):
    env = UnitreeEnv()
    env.max_steps = 500

    print("加载 PPO 模型...")
    ppo_model = PPO.load("models/unitree_robust")

    all_obs = []
    all_action_chunks = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_obs = []
        episode_actions = []

        for step in range(500):
            action, _ = ppo_model.predict(obs, deterministic=True)
            action = np.clip(action, -0.5, 0.5)
            episode_obs.append(obs.copy())
            episode_actions.append(action.copy())

            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        # 生成 action chunk
        T = len(episode_actions)
        for i in range(T - chunk_size):
            all_obs.append(episode_obs[i])
            all_action_chunks.append(episode_actions[i:i + chunk_size])

        if (ep + 1) % 20 == 0:
            print(f"已完成 {ep + 1}/{n_episodes} 条轨迹")

    obs_array = np.array(all_obs)
    action_chunk_array = np.array(all_action_chunks)  # (N, 16, 12)

    obs_mean = obs_array.mean(axis=0)
    obs_std = obs_array.std(axis=0) + 1e-6
    obs_normalized = (obs_array - obs_mean) / obs_std
    obs_normalized = np.clip(obs_normalized, -5.0, 5.0)

    with h5py.File(save_path, 'w') as f:
        f.create_dataset("obs", data=obs_normalized)
        f.create_dataset("action_chunk", data=action_chunk_array)
        f.create_dataset("obs_mean", data=obs_mean)
        f.create_dataset("obs_std", data=obs_std)

    print(f"\n✅ 数据已保存：{save_path}")
    print(f"   总样本数：{len(obs_normalized)}")
    print(f"   action_chunk 形状：{action_chunk_array.shape}")


if __name__ == "__main__":
    collect()