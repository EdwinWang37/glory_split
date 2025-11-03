import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataload.vae_generator import FakeNewsGenerator

def visualize_vectors(original_vectors, anti_vectors, uniform_vectors, title="Vector Distribution", filename="vector_distribution_comparison.png"):
    """使用PCA将向量降维到2D并可视化分布"""
    # 将所有向量合并在一起进行PCA
    all_vectors = torch.cat([original_vectors, anti_vectors, uniform_vectors], dim=0).cpu().numpy()
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(all_vectors)
    
    # 分离不同类型的向量
    n_original = len(original_vectors)
    n_anti = len(anti_vectors)
    original_2d = vectors_2d[:n_original]
    anti_2d = vectors_2d[n_original:n_original+n_anti]
    uniform_2d = vectors_2d[n_original+n_anti:]
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(original_2d[:, 0], original_2d[:, 1], c='blue', label='Original Vectors', alpha=0.6)
    plt.scatter(anti_2d[:, 0], anti_2d[:, 1], c='red', label='Anti-interest Vectors', alpha=0.6)
    plt.scatter(uniform_2d[:, 0], uniform_2d[:, 1], c='green', label='Uniform Vectors', alpha=0.6)
    
    plt.title(title)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def compute_cosine_similarity_stats(vectors):
    """计算向量集合中的平均余弦相似度"""
    # 计算向量之间的余弦相似度
    normalized_vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)
    similarity_matrix = torch.mm(normalized_vectors, normalized_vectors.t())
    
    # 移除对角线上的自相似度
    mask = torch.ones_like(similarity_matrix) - torch.eye(len(vectors), device=vectors.device)
    similarities = similarity_matrix[mask.bool()]
    
    return {
        'mean': similarities.mean().item(),
        'std': similarities.std().item(),
        'min': similarities.min().item(),
        'max': similarities.max().item()
    }

def main():
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 初始化参数
    input_dim = 400  # 新闻向量维度
    latent_dim = 50  # VAE潜在空间维度
    n_original = 50  # 原始向量数量
    n_generate = 50  # 生成向量数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模拟的原始新闻向量（假设它们在某个区域聚集）
    mean_vector = torch.randn(input_dim, device=device)
    original_vectors = mean_vector + 0.5 * torch.randn(n_original, input_dim, device=device)
    
    # 初始化并训练VAE
    generator = FakeNewsGenerator(input_dim=input_dim, latent_dim=latent_dim, device=device)
    generator.train_vae(original_vectors, epochs=1000)
    
    # 生成两种不同类型的向量
    anti_vectors = generator.generate_fake_news(original_vectors.mean(dim=0), num_samples=n_generate)
    dummy_user_real_vectors = torch.randn(2, input_dim, device=device)
    uniform_vectors_adversarial = generator.generate_uniform_space_filling_vectors(user_real_vectors=dummy_user_real_vectors, num_fake_per_user=n_generate)
    uniform_vectors_enhanced = generator.generate_uniform_space_filling_vectors(user_real_vectors=dummy_user_real_vectors, num_fake_per_user=n_generate)
    
    # 可视化三种向量的分布
    visualize_vectors(original_vectors, anti_vectors, uniform_vectors_adversarial, title="Vector Distribution (Adversarial Uniform)", filename="vector_distribution_adversarial.png")
    
    # 可视化增强VAE生成的均匀向量
    visualize_vectors(original_vectors, anti_vectors, uniform_vectors_enhanced, title="Vector Distribution (Enhanced VAE Uniform)", filename="vector_distribution_enhanced.png")
    
    # 计算并打印相似度统计信息
    print("\nOriginal vectors similarity stats:")
    print(compute_cosine_similarity_stats(original_vectors))
    
    print("\nAnti-interest vectors similarity stats:")
    print(compute_cosine_similarity_stats(anti_vectors))
    
    print("\nUniform vectors (adversarial) similarity stats:")
    print(compute_cosine_similarity_stats(uniform_vectors_adversarial))
    
    print("\nUniform vectors (enhanced VAE) similarity stats:")
    print(compute_cosine_similarity_stats(uniform_vectors_enhanced))
    
    # 计算不同类型向量之间的平均相似度
    original_norm = original_vectors / torch.norm(original_vectors, dim=1, keepdim=True)
    anti_norm = anti_vectors / torch.norm(anti_vectors, dim=1, keepdim=True)
    uniform_adversarial_norm = uniform_vectors_adversarial / torch.norm(uniform_vectors_adversarial, dim=1, keepdim=True)
    uniform_enhanced_norm = uniform_vectors_enhanced / torch.norm(uniform_vectors_enhanced, dim=1, keepdim=True)
    
    print("\nMean similarity between original and anti-interest vectors:")
    print(torch.mm(original_norm, anti_norm.t()).mean().item())
    
    print("\nMean similarity between original and uniform vectors (adversarial):")
    print(torch.mm(original_norm, uniform_adversarial_norm.t()).mean().item())
    
    print("\nMean similarity between original and uniform vectors (enhanced VAE):")
    print(torch.mm(original_norm, uniform_enhanced_norm.t()).mean().item())

if __name__ == '__main__':
    main()