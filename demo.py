from stable_baselines3 import PPO
from env.unitree_env import UnitreeEnv

model = PPO.load('models/unitree_final_optimized')
env = UnitreeEnv()
env.max_steps = 500

obs, _ = env.reset()
start_x = env.data.qpos[0]

print("开始演示（500 步）...")

for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    if step % 100 == 0:
        distance = env.data.qpos[0] - start_x
        print(f"步 {step}: 前进 {distance:.2f} 米")

    if terminated or truncated:
        break

total = env.data.qpos[0] - start_x
print(f"\n✅ 演示完成，总前进距离: {total:.2f} 米")