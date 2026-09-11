from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from env.unitree_env import UnitreeEnv
import os

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

env = DummyVecEnv([lambda: UnitreeEnv()])
eval_env = DummyVecEnv([lambda: UnitreeEnv()])

model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.02,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    tensorboard_log="./logs/"
)

callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/best/',
    log_path='./logs/',
    eval_freq=5000,
    deterministic=True,
    n_eval_episodes=5
)

print("开始训练（30万步）...")
model.learn(total_timesteps=300000, callback=callback)
model.save('models/unitree_final_optimized')
print("✅ 训练完成，模型已保存到 models/unitree_final_optimized.zip")