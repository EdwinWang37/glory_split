import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 模拟新闻向量和用户兴趣向量
def generate_data(num_samples=20, input_dim=100):
    # 创建一个中心点，新闻向量将围绕这个中心点分布
    center = torch.randn(1, input_dim)
    
    # 生成围绕中心点的新闻向量，使用较小的标准差使其聚集
    news_vectors = center + torch.randn(num_samples, input_dim) * 0.05  # 0.几是聚集程度，可以调整
    
    # 将数据归一化到0-1范围
    #news_vectors = (news_vectors - news_vectors.min()) / (news_vectors.max() - news_vectors.min())
    news_vectors = torch.clamp(center + torch.randn(num_samples, input_dim) * 0.1, 0, 1)
    
    # 用户兴趣向量设为新闻向量的均值（表示用户对这类新闻感兴趣）
    user_interest = torch.mean(news_vectors, dim=0, keepdim=True)
    
    return news_vectors, user_interest


# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # 均值
        self.fc22 = nn.Linear(400, latent_dim)  # 方差
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # 使用Sigmoid限制输出范围 [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# VAE损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# 训练VAE模型
def train_vae(model, news_vectors, epochs=10000):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(news_vectors)
        loss = loss_function(recon_batch, news_vectors, mu, logvar)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    return model


# 生成用户兴趣向量的“相反”向量
def generate_anti_interest_vector(model, user_interest, latent_dim, num_samples, num_iterations=1000):
    # 先将用户兴趣向量编码到潜在空间
    model.eval()
    with torch.no_grad():
        user_mu, user_logvar = model.encode(user_interest)
        user_latent = model.reparameterize(user_mu, user_logvar)

    # 初始化多个与用户兴趣相反的潜在向量
    anti_latent = torch.nn.Parameter(torch.randn(num_samples, latent_dim))
    optimizer = optim.Adam([anti_latent], lr=0.01)

    # 在 generate_anti_interest_vector 函数中修改损失函数
    for step in range(num_iterations):
        optimizer.zero_grad()
        generated_news = model.decode(anti_latent)
    
        # 计算余弦相似度
        similarity = F.cosine_similarity(generated_news, user_interest.expand_as(generated_news), dim=1)
    
        # 计算欧几里得距离（负值，因为我们要最大化距离）
        distance = -torch.norm(generated_news - user_interest.expand_as(generated_news), dim=1)
    
        # 组合损失：同时最小化相似度和最大化距离
        loss = -similarity.mean() + distance.mean() * 0.1  # 可以调整权重
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step [{step}/{num_iterations}], Average Similarity: {similarity.mean().item():.4f}")

    return model.decode(anti_latent).detach()


# 主程序
def main():
    # 初始化数据
    num_samples = 20
    input_dim = 100
    latent_dim = 5
    news_vectors, user_interest = generate_data(num_samples, input_dim)

    # 初始化并训练VAE模型
    vae = VAE(input_dim, latent_dim)
    vae = train_vae(vae, news_vectors)

    # 生成相反兴趣的新闻向量（生成与原始新闻向量相同数量）
    generated_anti_news = generate_anti_interest_vector(vae, user_interest, latent_dim, num_samples)

    # 计算原始用户兴趣与生成新闻的相似度
    original_similarity = F.cosine_similarity(user_interest, news_vectors, dim=1).mean()
    anti_similarity = F.cosine_similarity(user_interest.expand_as(generated_anti_news), generated_anti_news, dim=1).mean()

    print(f"原始新闻与用户兴趣的平均相似度: {original_similarity:.4f}")
    print(f"生成新闻与用户兴趣的平均相似度: {anti_similarity:.4f}")

    # 可视化生成的新闻向量 - 使用降维方法
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import numpy as np
    
    # 合并所有向量用于降维
    all_vectors = torch.cat([
        news_vectors,  # 原始新闻向量
        generated_anti_news,  # 生成的假新闻向量（现在是多个）
        user_interest  # 用户兴趣向量
    ], dim=0)
    
    # 转换为numpy数组
    all_vectors_np = all_vectors.detach().numpy()
    
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(all_vectors_np)
    
    # 分离不同类型的向量
    original_news_2d = vectors_2d[:num_samples]  # 原始新闻向量
    generated_news_2d = vectors_2d[num_samples:num_samples*2]  # 生成的假新闻向量（现在是多个）
    user_interest_2d = vectors_2d[-1:]  # 用户兴趣向量
    
    # 创建散点图
    plt.figure(figsize=(12, 10))
    
    # 第一个子图：PCA降维结果
    plt.subplot(2, 2, 1)
    plt.scatter(original_news_2d[:, 0], original_news_2d[:, 1], 
                c='blue', alpha=0.6, label='general', s=50)
    plt.scatter(generated_news_2d[:, 0], generated_news_2d[:, 1], 
                c='red', marker='s', label='fake', s=50, alpha=0.7)
    plt.scatter(user_interest_2d[:, 0], user_interest_2d[:, 1], 
                c='green', marker='^', label='user_interest', s=100)
    plt.xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA降维可视化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 第二个子图：t-SNE降维结果（如果数据量足够）
    if len(all_vectors_np) >= 4:  # t-SNE需要至少4个样本
        plt.subplot(2, 2, 2)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(3, len(all_vectors_np)-1))
        vectors_tsne = tsne.fit_transform(all_vectors_np)
        
        original_news_tsne = vectors_tsne[:num_samples]
        generated_news_tsne = vectors_tsne[num_samples:num_samples*2]
        user_interest_tsne = vectors_tsne[-1:]
        
        plt.scatter(original_news_tsne[:, 0], original_news_tsne[:, 1], 
                    c='blue', alpha=0.6, label='general', s=50)
        plt.scatter(generated_news_tsne[:, 0], generated_news_tsne[:, 1], 
                    c='red', marker='s', label='fake', s=100)
        plt.scatter(user_interest_tsne[:, 0], user_interest_tsne[:, 1], 
                    c='green', marker='^', label='user_interest', s=100)
        plt.xlabel('t-SNE 维度 1')
        plt.ylabel('t-SNE 维度 2')
        plt.title('t-SNE降维可视化')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 第三个子图：距离分析
    plt.subplot(2, 2, 3)
    # 计算用户兴趣向量到各个新闻向量的距离
    distances_to_original = torch.norm(news_vectors - user_interest, dim=1).detach().numpy()
    distances_to_generated = torch.norm(generated_anti_news - user_interest.expand_as(generated_anti_news), dim=1).detach().numpy()
    
    plt.hist(distances_to_original, bins=10, alpha=0.7, label='distance to general', color='blue')
    plt.hist(distances_to_generated, bins=10, alpha=0.7, label='distance to fake', color='red')
    plt.xlabel('oujilide‘s distance')
    plt.ylabel('frequency')
    plt.title('user’s interest vector to some news vector')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 第四个子图：相似度分析
    plt.subplot(2, 2, 4)
    similarities_to_original = F.cosine_similarity(user_interest, news_vectors, dim=1).detach().numpy()
    similarities_to_generated = F.cosine_similarity(user_interest.expand_as(generated_anti_news), generated_anti_news, dim=1).detach().numpy()
    
    plt.hist(similarities_to_original, bins=10, alpha=0.7, label='similarity to general', color='blue')
    plt.hist(similarities_to_generated, bins=10, alpha=0.7, label='similarity to fake', color='red')
    plt.xlabel('cos similarity')
    plt.ylabel('frequency')
    plt.title('The similarity distribution between user interest vectors and news vectors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 保存图片到文件
    plt.savefig('interest_vectors_analysis.png', dpi=300, bbox_inches='tight')
    print("图片已保存为 'interest_vectors_analysis.png'")
    
    # 打印一些统计信息
    print(f"\n=== 降维分析结果 ===")
    print(f"PCA解释的总方差: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"原始新闻向量与用户兴趣的平均相似度: {similarities_to_original.mean():.4f}")
    print(f"生成新闻向量与用户兴趣的平均相似度: {similarities_to_generated.mean():.4f}")
    print(f"原始新闻向量到用户兴趣的平均距离: {distances_to_original.mean():.4f}")
    print(f"生成新闻向量到用户兴趣的平均距离: {distances_to_generated.mean():.4f}")
    # 删除重复的第240行，或者如果需要显示单个距离，可以改为：
    # print(f"第一个生成新闻向量到用户兴趣的距离: {distances_to_generated[0]:.4f}")
    plt.show()  # 在远程服务器上无法显示图形界面
    # 保存图片到文件
    plt.savefig('interest_vectors_comparison.png')
    print("图片已保存为 'interest_vectors_comparison.png'")


if __name__ == "__main__":
    main()