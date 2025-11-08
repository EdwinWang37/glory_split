#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比 uniform_space_filling 与 anti_interest 两种假新闻生成策略，使用真实数据计算指标。

运行示例：
python scripts/compare_vae_strategies.py \
  --config configs/uniform_vae_config.yaml \
  --save-plot results/vae_analysis/strategy_compare.png \
  --fake-per-user 5 --max-users 500
"""

import os
import sys
import argparse
from types import SimpleNamespace
from pathlib import Path
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataload.vae_fake_news_generator import IntegratedFakeNewsGenerator


def load_cfg(config_path: str) -> SimpleNamespace:
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    # 兼容两种配置结构：平铺和分组
    ns = SimpleNamespace()
    # 顶层字段
    for k, v in raw.items():
        if isinstance(v, dict):
            setattr(ns, k, SimpleNamespace(**v))
        else:
            setattr(ns, k, v)

    # 回退字段名映射（不同配置里的命名差异）
    if getattr(ns, 'use_vae_for_fake_news', None) is None and hasattr(ns, 'vae'):
        ns.use_vae_for_fake_news = ns.vae.get('use_vae_for_fake_news', True)
    if getattr(ns, 'vae_latent_dim', None) is None and hasattr(ns, 'vae'):
        ns.vae_latent_dim = ns.vae.get('latent_dim', 50)
    if getattr(ns, 'vae_training_epochs', None) is None and hasattr(ns, 'vae'):
        ns.vae_training_epochs = ns.vae.get('training_epochs', 1000)
    if getattr(ns, 'vae_generation_iterations', None) is None and hasattr(ns, 'vae'):
        ns.vae_generation_iterations = ns.vae.get('generation_iterations', 500)
    if getattr(ns, 'fake_news_per_user', None) is None and hasattr(ns, 'vae'):
        ns.fake_news_per_user = ns.vae.get('fake_news_per_user', 10)
    if getattr(ns, 'pca_components', None) is None:
        ns.pca_components = getattr(getattr(ns, 'evaluation', SimpleNamespace()), 'pca_components', 50)

    # 数据集路径
    if not hasattr(ns, 'dataset'):
        raise ValueError('config 缺少 dataset 字段，至少需要 dataset.train_dir')
    if not hasattr(ns.dataset, 'train_dir'):
        raise ValueError('config.dataset 需要包含 train_dir')

    # 模型字段
    if not hasattr(ns, 'model'):
        ns.model = SimpleNamespace(title_size=30)
    if not hasattr(ns.model, 'title_size'):
        ns.model.title_size = 30

    return ns


def collect_user_histories(train_dir: str, max_users: int = None):
    behaviors_path = Path(train_dir) / 'behaviors.tsv'
    if not behaviors_path.exists():
        raise FileNotFoundError(f'未找到 {behaviors_path}')

    user_histories = {}
    with open(behaviors_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            iid, uid, time, history = parts[:4]
            if uid not in user_histories:
                user_histories[uid] = []
            if history.strip():
                user_histories[uid].extend(history.split())

    # 去重并截断
    for uid in list(user_histories.keys()):
        user_histories[uid] = list(set(user_histories[uid]))
    if max_users is not None and len(user_histories) > max_users:
        # 简单截断前 max_users 个用户
        keys = list(user_histories.keys())[:max_users]
        user_histories = {k: user_histories[k] for k in keys}

    return user_histories


def compute_cosine_similarity_stats(tensor: torch.Tensor):
    if tensor.ndim != 2:
        raise ValueError('输入需为二维矩阵 [N, D]')
    normalized = tensor / torch.norm(tensor, dim=1, keepdim=True).clamp(min=1e-8)
    sim = torch.mm(normalized, normalized.t())
    mask = torch.ones_like(sim) - torch.eye(len(tensor), device=sim.device)
    vals = sim[mask.bool()]
    return {
        'mean': vals.mean().item(),
        'std': vals.std().item(),
        'min': vals.min().item(),
        'max': vals.max().item(),
    }


def plot_pca(real_vectors: torch.Tensor, anti_vectors: torch.Tensor, uniform_vectors: torch.Tensor, save_path: str):
    all_np = torch.cat([real_vectors, anti_vectors, uniform_vectors], dim=0).cpu().numpy()
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_np)
    n_real = len(real_vectors)
    n_anti = len(anti_vectors)
    real_2d = proj[:n_real]
    anti_2d = proj[n_real:n_real + n_anti]
    uniform_2d = proj[n_real + n_anti:]

    plt.figure(figsize=(10, 8))
    plt.scatter(real_2d[:, 0], real_2d[:, 1], c='blue', label='Real', alpha=0.5, s=10)
    plt.scatter(anti_2d[:, 0], anti_2d[:, 1], c='red', label='Anti-interest', alpha=0.6, s=12)
    plt.scatter(uniform_2d[:, 0], uniform_2d[:, 1], c='green', label='Uniform Space Filling', alpha=0.6, s=12)
    plt.title('PCA Comparison: Real vs Anti vs Uniform')
    plt.legend()
    plt.grid(True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare fake news generation strategies on real data')
    parser.add_argument('--config', type=str, default='configs/uniform_vae_config.yaml', help='配置文件路径')
    parser.add_argument('--save-plot', type=str, default='', help='保存PCA对比图的路径（可选）')
    parser.add_argument('--fake-per-user', type=int, default=None, help='每用户生成的假新闻数量（覆盖配置）')
    parser.add_argument('--max-users', type=int, default=500, help='参与对比的最大用户数')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.fake_per_user is not None:
        cfg.fake_news_per_user = args.fake_per_user

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = IntegratedFakeNewsGenerator(cfg, device=device)

    # generator 期望 data_dir 为 train/val/test 的父目录
    data_root = str(Path(cfg.dataset.train_dir).parent)
    ok = gen.initialize_vae(data_dir=data_root)
    if not ok:
        print('VAE 初始化失败，退出')
        sys.exit(1)

    # 收集用户历史
    user_histories = collect_user_histories(cfg.dataset.train_dir, max_users=args.max_users)
    print(f'参与对比的用户数: {len(user_histories)}')

    # 生成 anti_interest
    anti_dict = gen.generate_personalized_fake_news(user_histories, num_fake_per_user=cfg.fake_news_per_user, generation_strategy='anti_interest')
    anti_vectors = torch.stack([v for v in anti_dict.values()])

    # 生成 uniform_space_filling
    uniform_dict = gen.generate_personalized_fake_news(user_histories, num_fake_per_user=cfg.fake_news_per_user, generation_strategy='uniform_space_filling')
    uniform_vectors = torch.stack([v for v in uniform_dict.values()])

    # 真实向量集合
    real_vectors = gen.real_news_vectors

    # 指标计算
    print('\n=== Cosine similarity (within set) ===')
    print('Anti-interest:', compute_cosine_similarity_stats(anti_vectors))
    print('Uniform space filling:', compute_cosine_similarity_stats(uniform_vectors))

    # 与真实向量的平均相似度
    def mean_cross_similarity(a: torch.Tensor, b: torch.Tensor):
        a_n = a / torch.norm(a, dim=1, keepdim=True).clamp(min=1e-8)
        b_n = b / torch.norm(b, dim=1, keepdim=True).clamp(min=1e-8)
        return torch.mm(a_n, b_n.t()).mean().item()

    # 为避免爆内存，随机采样真实集合
    sample_n = min(5000, len(real_vectors))
    perm = torch.randperm(len(real_vectors))[:sample_n]
    real_sample = real_vectors[perm]

    print('\n=== Mean similarity to real news sample ===')
    print('Anti-interest -> Real:', mean_cross_similarity(anti_vectors, real_sample))
    print('Uniform -> Real:', mean_cross_similarity(uniform_vectors, real_sample))

    # 均匀性验证
    print('\n=== Uniformity metrics (PCA-based) ===')
    anti_uniform = gen.validate_distribution_uniformity(real_sample, anti_vectors, pca_components=getattr(cfg, 'pca_components', 50))
    uniform_uniform = gen.validate_distribution_uniformity(real_sample, uniform_vectors, pca_components=getattr(cfg, 'pca_components', 50))
    print('Anti-interest:', anti_uniform)
    print('Uniform space filling:', uniform_uniform)

    # 可视化
    if args.save_plot:
        plot_pca(real_sample, anti_vectors, uniform_vectors, args.save_plot)
        print(f'PCA对比图已保存到 {args.save_plot}')


if __name__ == '__main__':
    main()