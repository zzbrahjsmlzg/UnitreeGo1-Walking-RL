import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from stable_baselines3 import PPO
from env.unitree_env import UnitreeEnv

# 加载模型
model = PPO.load("models/unitree_robust")


def run_test(name, perturb_fn):
    """跑一次测试，输出总距离"""
    env = UnitreeEnv()
    env.max_steps = 1000

    obs, _ = env.reset()
    perturb_fn(env)  # 施加扰动

    start_x = env.data.qpos[0]
    total_reward = 0

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    distance = env.data.qpos[0] - start_x
    print(f"[{name}] 距离: {distance:.2f} 米, 奖励: {total_reward:.2f}")
    return distance


# 基准（无扰动）
def no_perturb(env):
    pass


# 质量扰动
def mass_perturb(env):
    env.model.body_mass[:] *= 1.1


# 摩擦扰动
def friction_perturb(env):
    env.model.geom_friction[:, 0] *= 0.7


# 控制延迟
def delay_perturb(env):
    # 这个需要改 step()，先跳过
    pass


if __name__ == "__main__":
    print("=" * 50)
    print("Sim-to-Sim 鲁棒性测试")
    print("=" * 50)

    base = run_test("基准", no_perturb)
    m = run_test("质量+10%", mass_perturb)
    f = run_test("摩擦-30%", friction_perturb)

    print("\n结果对比：")
    print(f"  基准:     {base:.2f} 米")
    print(f"  质量+10%: {m:.2f} 米 ({m/base*100:.0f}%)")
    print(f"  摩擦-30%: {f:.2f} 米 ({f/base*100:.0f}%)")