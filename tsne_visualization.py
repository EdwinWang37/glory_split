#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t-SNE可视化脚本：为至少点击过30个新闻的用户生成新闻向量的t-SNE可视化图
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

# 添加src目录到路径
sys.path.append('src')
from models.component.news_encoder import NewsEncoder
from models.base.layers import *

# 导入NewsVectorizer
sys.path.append('.')
from load_news_encoder import NewsVectorizer

def load_model(model_path):
    """
    加载预训练的新闻编码器模型
    """
    print(f"Loading model from: {model_path}")
    
    # 使用NewsVectorizer来加载模型
    vectorizer = NewsVectorizer(model_path)
    
    return vectorizer.news_encoder, vectorizer.device

def load_user_behavior_data(file_path):
    """
    加载用户行为数据
    """
    print(f"加载用户行为数据: {file_path}")
    
    user_clicks = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                user_id = parts[1]
                clicked_news = parts[3]
                
                if clicked_news and clicked_news != '':
                    news_list = clicked_news.split(' ')
                    user_clicks[user_id].extend(news_list)
    
    print(f"总共有 {len(user_clicks)} 个用户")
    return user_clicks

def find_users_with_min_clicks(user_clicks, min_clicks=30, max_clicks=50, num_users=10):
    """
    找到点击数在指定范围内的用户
    """
    qualified_users = []
    
    for user_id, clicks in user_clicks.items():
        unique_clicks = list(set(clicks))  # 去重
        if min_clicks <= len(unique_clicks) <= max_clicks:
            qualified_users.append((user_id, len(unique_clicks), unique_clicks))
    
    # 按点击数量排序
    qualified_users.sort(key=lambda x: x[1], reverse=True)
    
    print(f"找到 {len(qualified_users)} 个用户的点击数在 {min_clicks}-{max_clicks} 之间")
    
    if len(qualified_users) < num_users:
        print(f"警告：只找到 {len(qualified_users)} 个符合条件的用户，少于要求的 {num_users} 个")
        return qualified_users
    
    selected_users = qualified_users[:num_users]
    for i, (user_id, click_count, _) in enumerate(selected_users):
        print(f"用户 {i+1}: {user_id}, 点击了 {click_count} 个不同新闻")
    
    return selected_users

def load_real_news_data():
    """
    加载真实新闻数据
    """
    news_file = 'train/nltk_token_news.bin'
    news_dict_file = 'train/news_dict.bin'
    
    try:
        print(f"Loading real news data from: {news_file}")
        with open(news_file, 'rb') as f:
            news_input = pickle.load(f)
        
        print(f"Loading news dictionary from: {news_dict_file}")
        with open(news_dict_file, 'rb') as f:
            news_dict = pickle.load(f)
        
        # 加载所有新闻数据
        news_data = []
        for i in range(len(news_input)):
            news_data.append(news_input[i].tolist())
        
        print(f"Successfully loaded all {len(news_data)} news articles")
        print(f"加载了 {len(news_dict)} 条新闻的ID映射")
        
        return news_data, news_dict
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

def generate_news_vectors_for_users(selected_users, vectorizer, news_data, news_dict):
    """
    为选中的用户生成新闻向量
    """
    all_vectors = []
    all_labels = []
    user_info = []
    
    for user_idx, (user_id, click_count, clicked_news) in enumerate(selected_users):
        print(f"\n=== 处理用户 {user_idx + 1}: {user_id} ===")
        print(f"该用户点击了 {len(clicked_news)} 个不同新闻")
        
        user_vectors = []
        valid_news_count = 0
        
        for news_id in clicked_news:
            if news_id in news_dict:
                news_index = news_dict[news_id]
                
                if news_index < len(news_data):
                    # 获取新闻的token IDs
                    news_tokens = news_data[news_index]
                    
                    # 使用vectorizer生成向量
                    try:
                        news_vector = vectorizer.vectorize_news(news_tokens)
                        if news_vector is not None:
                            news_vector_np = news_vector.cpu().numpy().flatten()
                            user_vectors.append(news_vector_np)
                            valid_news_count += 1
                    except Exception as e:
                        print(f"生成新闻 {news_id} 向量时出错: {e}")
                        continue
        
        print(f"成功生成了 {valid_news_count} 个新闻向量")
        
        if user_vectors:
            all_vectors.extend(user_vectors)
            all_labels.extend([f"用户{user_idx + 1}" for _ in range(len(user_vectors))])
            user_info.append((user_id, len(user_vectors)))
    
    return np.array(all_vectors), all_labels, user_info

def create_tsne_pca_visualization(vectors, labels, user_info, output_file='tsne_pca_news_vectors.png'):
    """
    为每个用户单独创建t-SNE和PCA对比可视化图
    """
    print(f"\n=== 创建t-SNE和PCA对比可视化 ===")
    print(f"向量形状: {vectors.shape}")
    
    # 为每个用户分配不同颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    unique_labels = list(set(labels))
    
    for i, user_label in enumerate(unique_labels):
        mask = np.array(labels) == user_label
        user_vectors = vectors[mask]
        
        if len(user_vectors) < 2:
            print(f"{user_label} 的新闻向量数量不足，跳过可视化")
            continue
            
        print(f"为 {user_label} 创建t-SNE和PCA对比可视化，处理 {len(user_vectors)} 个新闻向量...")
        
        # 执行t-SNE降维
        perplexity = min(30, len(user_vectors) - 1)
        if perplexity < 5:
            perplexity = min(5, len(user_vectors) - 1)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_vectors_2d = tsne.fit_transform(user_vectors)
        
        # 执行PCA降维
        pca = PCA(n_components=2, random_state=42)
        pca_vectors_2d = pca.fit_transform(user_vectors)
        
        # 创建包含两个子图的可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # t-SNE子图
        ax1.scatter(tsne_vectors_2d[:, 0], tsne_vectors_2d[:, 1], 
                   c=colors[i % len(colors)], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        ax1.set_title(f'{user_label} - t-SNE降维', fontsize=14)
        ax1.set_xlabel('t-SNE 维度 1', fontsize=12)
        ax1.set_ylabel('t-SNE 维度 2', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # PCA子图
        ax2.scatter(pca_vectors_2d[:, 0], pca_vectors_2d[:, 1], 
                   c=colors[i % len(colors)], alpha=0.7, s=60, edgecolors='black', linewidth=0.5, marker='s')
        ax2.set_title(f'{user_label} - PCA降维', fontsize=14)
        ax2.set_xlabel(f'PCA 主成分 1 (解释方差: {pca.explained_variance_ratio_[0]:.3f})', fontsize=12)
        ax2.set_ylabel(f'PCA 主成分 2 (解释方差: {pca.explained_variance_ratio_[1]:.3f})', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        user_info_item = next((info for info in user_info if user_label.endswith(str(user_info.index(info) + 1))), None)
        if user_info_item:
            stats_text = f"新闻向量数: {len(user_vectors)}\n向量维度: {user_vectors.shape[1]}\n用户ID: {user_info_item[0]}"
        else:
            stats_text = f"新闻向量数: {len(user_vectors)}\n向量维度: {user_vectors.shape[1]}"
        
        # 在整个图的左上角添加统计信息
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 添加总标题
        fig.suptitle(f'{user_label} 新闻向量降维对比 (圆点=t-SNE, 方块=PCA)', fontsize=16)
        
        plt.tight_layout()
        filename = f'tsne_pca_{user_label.replace(" ", "_")}_vectors.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{user_label} 的t-SNE和PCA对比可视化图已保存到 {filename}")
    
    return None

def main():
    """
    主函数
    """
    print("=== 开始t-SNE新闻向量可视化 ===")
    
    # 1. 加载模型
    model_path = 'GLORY_MINDsmall_default_auc0.6760649681091309.pth'
    vectorizer = NewsVectorizer(model_path)
    
    # 2. 加载用户行为数据
    behavior_file = 'train/behaviors_np4_0.tsv'
    user_clicks = load_user_behavior_data(behavior_file)
    
    # 3. 找到符合条件的用户
    selected_users = find_users_with_min_clicks(user_clicks, min_clicks=30, max_clicks=50, num_users=10)
    
    if not selected_users:
        print("没有找到符合条件的用户")
        return
    
    # 4. 加载新闻数据
    news_data, news_dict = load_real_news_data()
    if news_data is None:
        print("无法加载新闻数据")
        return
    
    # 5. 生成新闻向量
    vectors, labels, user_info = generate_news_vectors_for_users(
        selected_users, vectorizer, news_data, news_dict
    )
    
    if len(vectors) == 0:
        print("没有生成任何向量")
        return
    
    # 6. 创建t-SNE可视化
    vectors_2d = create_tsne_visualization(vectors, labels, user_info)
    
    print("\n=== 完成 ===")
    print(f"总共处理了 {len(vectors)} 个新闻向量")
    print(f"来自 {len(set(labels))} 个用户")

if __name__ == "__main__":
    main()