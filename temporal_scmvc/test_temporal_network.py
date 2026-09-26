from dataloader_mhealth_temporal import load_mhealth_temporal
from network_temporal import NetworkTemporal
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_mhealth_temporal(
    window_size=50,
    stride=25,
    max_samples=64
)

xs, y, idx = dataset[0]
print("Single sample:")
for i, x in enumerate(xs):
    print(f"View {i+1} shape: {x.shape}")

batch_xs = []
for v in range(view):
    stacked = torch.stack([dataset[i][0][v] for i in range(8)], dim=0).to(device)
    batch_xs.append(stacked)

model = NetworkTemporal(
    view=view,
    input_size=dims,
    feature_dim=64,
    high_feature_dim=20,
    device=device,
    seq_len=50
).to(device)

xrs, zs, rs, H = model(batch_xs)

print("\nModel output shapes:")
for i in range(view):
    print(f"xr[{i}] shape: {xrs[i].shape}")
    print(f"z[{i}] shape: {zs[i].shape}")
    print(f"r[{i}] shape: {rs[i].shape}")

print("H shape:", H.shape)