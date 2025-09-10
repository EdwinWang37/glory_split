
import torch
import torch.nn as nn

class NoiseAddition(nn.Module):
    def __init__(self, noise_std=0.1):
        super(NoiseAddition, self).__init__()
        self.noise_std = noise_std

    def forward(self, x):
        # 生成与输入相同形状的高斯噪声
        noise = torch.randn_like(x) * self.noise_std
        # 将噪声添加到输入上
        noisy_x = x + noise
        return noisy_x


# 去噪模块（MLP）
class DenoisingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(DenoisingMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, noisy_x):
        return self.layers(noisy_x)