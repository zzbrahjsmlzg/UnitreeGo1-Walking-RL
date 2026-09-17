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


class MLPPolicy(torch.nn.Module):
    def __init__(self, obs_dim=30, act_dim=12):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)


def train():
    dataset = Go1Dataset()
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    policy = MLPPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print("=" * 50)
    print("开始训练 MLP 行为克隆")
    print("=" * 50)

    for epoch in range(30):
        total_loss = 0
        for obs, action in loader:
            obs, action = obs.to(device), action.to(device)
            pred = policy(obs)
            loss = loss_fn(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1:2d}: loss = {avg_loss:.6f}")

    torch.save(policy.state_dict(), "imitation/mlp_policy.pt")
    print("\n✅ 模型已保存：imitation/mlp_policy.pt")


if __name__ == "__main__":
    train()