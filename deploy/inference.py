import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from env.unitree_env import UnitreeEnv

model = PPO.load("models/unitree_final_optimized")
env = UnitreeEnv()
env.max_steps = 1000

obs, _ = env.reset()
start_x = env.data.qpos[0]

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    if step % 100 == 0:
        distance = env.data.qpos[0] - start_x
        print(f"步 {step}: 距离 {distance:.2f} 米")

    if terminated or truncated:
        break

print(f"\n总距离: {env.data.qpos[0] - start_x:.2f} 米")