import h5py
import numpy as np

with h5py.File("imitation/go1_data.h5", "r") as f:
    obs = f["obs"][:]
    action = f["action"][:]

print("=" * 50)
print("数据检查结果")
print("=" * 50)

print(f"obs 形状: {obs.shape}")
print(f"action 形状: {action.shape}")
print(f"obs dtype: {obs.dtype}")
print(f"action dtype: {action.dtype}")

print(f"\nobs 范围: [{obs.min():.4f}, {obs.max():.4f}]")
print(f"action 范围: [{action.min():.4f}, {action.max():.4f}]")

print(f"\nobs 有 NaN 吗: {np.isnan(obs).any()}")
print(f"action 有 NaN 吗: {np.isnan(action).any()}")

print(f"\nobs 均值 (前 5 维): {obs.mean(axis=0)[:5]}")
print(f"action 均值 (前 5 维): {action.mean(axis=0)[:5]}")