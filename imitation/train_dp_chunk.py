import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Go1ChunkDataset(Dataset):
    def __init__(self, path="imitation/go1_data_chunk.h5"):
        with h5py.File(path, 'r') as f:
            self.obs = torch.tensor(f["obs"][:], dtype=torch.float32)
            self.action_chunk = torch.tensor(f["action_chunk"][:], dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.action_chunk[idx]


class ChunkDiffusionPolicy(torch.nn.Module):
    def __init__(self, obs_dim=30, act_dim=12, chunk_size=16, hidden=256, T=10):
        super().__init__()
        self.T = T
        self.chunk_size = chunk_size
        self.act_flat = act_dim * chunk_size  # 16 * 12 = 192
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + self.act_flat + 1, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, self.act_flat)
        )

    def forward(self, obs, action_flat, t):
        t_norm = torch.full((obs.shape[0], 1), t / self.T, device=obs.device)
        x = torch.cat([obs, action_flat, t_norm], dim=-1)
        return self.net(x)


def train():
    dataset = Go1ChunkDataset()
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    policy = ChunkDiffusionPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print("=" * 50)
    print("训练带 Action Chunking 的 Diffusion Policy")
    print("=" * 50)

    for epoch in range(30):
        total_loss = 0
        for obs, action_chunk in loader:
            obs = obs.to(device)
            # 把 (B, 16, 12) 展平成 (B, 192)
            action_flat = action_chunk.reshape(action_chunk.shape[0], -1).to(device)

            t = np.random.randint(0, policy.T)
            noise = torch.randn_like(action_flat)
            alpha = 1.0 - t / policy.T
            action_noisy = alpha * action_flat + (1 - alpha) * noise

            pred_noise = policy(obs, action_noisy, t)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1:2d}: loss = {avg_loss:.6f}")

    torch.save(policy.state_dict(), "imitation/dp_chunk_policy.pt")
    print("\n✅ 模型已保存：imitation/dp_chunk_policy.pt")


if __name__ == "__main__":
    train()