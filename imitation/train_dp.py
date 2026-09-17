import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Go1Dataset(Dataset):
    def __init__(self, path="imitation/go1_data.h5"):
        with h5py.File(path, 'r') as f:
            self.obs = torch.tensor(f["obs"][:], dtype=torch.float32)
            self.action = torch.tensor(f["action"][:], dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.action[idx]


class SimpleDiffusionPolicy(torch.nn.Module):
    def __init__(self, obs_dim=30, act_dim=12, hidden=256, T=10):
        super().__init__()
        self.T = T  # 去噪步数
        # 输入：obs + action + timestep
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim + 1, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, act_dim)
        )

    def forward(self, obs, action_noisy, t):
        # t 归一化到 [0, 1]
        t_norm = torch.full((obs.shape[0], 1), t / self.T, device=obs.device)
        x = torch.cat([obs, action_noisy, t_norm], dim=-1)
        return self.net(x)


def train():
    dataset = Go1Dataset()
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    policy = SimpleDiffusionPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print("=" * 50)
    print("训练简化版 Diffusion Policy")
    print("=" * 50)

    for epoch in range(30):
        total_loss = 0
        for obs, action in loader:
            obs, action = obs.to(device), action.to(device)

            # 随机采样 timestep
            t = np.random.randint(0, policy.T)
            # 加噪声
            noise = torch.randn_like(action)
            alpha = 1.0 - t / policy.T
            action_noisy = alpha * action + (1 - alpha) * noise

            # 预测噪声
            pred_noise = policy(obs, action_noisy, t)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1:2d}: loss = {avg_loss:.6f}")

    torch.save(policy.state_dict(), "imitation/dp_policy.pt")
    print("\n✅ 模型已保存：imitation/dp_policy.pt")


if __name__ == "__main__":
    train()