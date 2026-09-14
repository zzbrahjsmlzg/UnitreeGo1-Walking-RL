import mujoco
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box

class UnitreeEnv(Env):
    def __init__(self, model_path='D:/mujoco_menagerie/unitree_go2/scene.xml'):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.action_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        
        self.step_count = 0
        self.max_steps = 500
        self.prev_x = 0.0
        self.prev_action = np.zeros(12)  # 初始化前一个动作
        
        # Go1 关节限位
        self.joint_limits = [
            (-1.2, 1.2), (-1.2, 1.2), (0.0, 2.5),
            (-1.2, 1.2), (-1.2, 1.2), (0.0, 2.5),
            (-1.2, 1.2), (-1.2, 1.2), (0.0, 2.5),
            (-1.2, 1.2), (-1.2, 1.2), (0.0, 2.5),
        ]

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_x = self.data.qpos[0]
        self.prev_action = np.zeros(12)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # 保存当前动作（用于平滑惩罚）
        self.action = action.copy()
        
        # 映射到关节角度
        for i in range(12):
            low, high = self.joint_limits[i]
            self.data.qpos[3 + i] = low + (action[i] + 1.0) / 2.0 * (high - low)
        
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._check_termination(obs)
        truncated = self.step_count >= self.max_steps
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        base_pos = self.data.qpos[0:3].copy()
        joint_pos = self.data.qpos[3:15].copy()
        base_vel = self.data.qvel[0:3].copy()
        joint_vel = self.data.qvel[3:15].copy()
        obs = np.concatenate([joint_pos, base_pos, joint_vel, base_vel])
        return obs.astype(np.float32)

    def _compute_reward(self, obs):
        height = obs[14]
        forward_vel = self.data.qvel[0]
        displacement = self.data.qpos[0] - self.prev_x
        self.prev_x = self.data.qpos[0]
        
        # 1. 位移奖励
        displacement_reward = displacement * 20.0
        
        # 2. 速度奖励
        speed_reward = forward_vel * 2.0
        
        # 3. 高度保持
        height_target = 0.45
        height_error = abs(height - height_target)
        if height_error > 0.1:
            height_penalty = (height_error - 0.1) * 3.0
        else:
            height_penalty = 0.0
        
        # 4. 颠簸惩罚
        vertical_vel = self.data.qvel[2]
        vertical_penalty = abs(vertical_vel) * 0.5
        
        # 5. 动作平滑惩罚
        if hasattr(self, 'action') and hasattr(self, 'prev_action'):
            action_diff = np.sum(np.abs(self.action - self.prev_action))
            smoothness_penalty = action_diff * 0.01
        else:
            smoothness_penalty = 0.0
        
        # 更新前一个动作
        if hasattr(self, 'action'):
            self.prev_action = self.action.copy()
        
        # 总奖励
        reward = (displacement_reward + speed_reward 
                  - height_penalty - vertical_penalty - smoothness_penalty)
        
        if height < 0.15:
            reward -= 10.0
        
        reward += 0.05
        return reward

    def _check_termination(self, obs):
        height = obs[14]
        return height < 0.1 or height > 1.0