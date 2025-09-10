import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        return self.fc4(h3)  # 移除sigmoid激活函数，允许输出任意实数值

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class FakeNewsGenerator:
    def __init__(self, input_dim, latent_dim=20, device='cuda'):
        self.device = device
        self.vae = VAE(input_dim, latent_dim).to(device)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.is_trained = False
        
    def generate_uniform_vectors(self, num_samples=50, method='adversarial'):
        """生成均匀分布的新闻向量
        
        Args:
            num_samples: 生成向量的数量
            method: 生成方法，'adversarial'为对抗训练生成，'enhanced_vae'为增强VAE生成
        """
        if not self.is_trained:
            raise ValueError("VAE模型尚未训练，请先调用train_vae方法")
            
        if method == 'adversarial':
            return self._generate_adversarial_uniform(num_samples)
        elif method == 'enhanced_vae':
            return self._generate_enhanced_vae_uniform(num_samples)
        else:
            raise ValueError("method参数必须是'adversarial'或'enhanced_vae'")
    
    def _generate_adversarial_uniform(self, num_samples):
        """使用对抗训练生成均匀分布向量"""
        self.vae.eval()
        
        # 初始化潜在向量
        z = torch.randn(num_samples, self.latent_dim, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=0.01)
        
        # 对抗训练：最大化生成向量之间的距离
        for iteration in range(200):
            optimizer.zero_grad()
            
            # 生成向量
            generated_vectors = self.vae.decode(z)
            
            # 计算向量间的成对距离
            pairwise_distances = torch.cdist(generated_vectors, generated_vectors, p=2)
            
            # 排除对角线元素（自身距离为0）
            mask = torch.eye(num_samples, device=self.device).bool()
            pairwise_distances = pairwise_distances.masked_fill(mask, float('inf'))
            
            # 最小化最小距离的负值（即最大化最小距离）
            min_distances = pairwise_distances.min(dim=1)[0]
            diversity_loss = -min_distances.mean()
            
            # 添加正则化项，防止向量过大
            regularization = 0.01 * torch.norm(generated_vectors, dim=1).mean()
            
            # 添加潜在空间正则化
            latent_reg = 0.001 * torch.norm(z, dim=1).mean()
            
            total_loss = diversity_loss + regularization + latent_reg
            total_loss.backward()
            optimizer.step()
            
            # 限制潜在向量的范围
            with torch.no_grad():
                z.clamp_(-3.0, 3.0)
        
        with torch.no_grad():
            final_vectors = self.vae.decode(z)
            
        return final_vectors.detach()
    
    def _generate_enhanced_vae_uniform(self, num_samples):
        """使用增强VAE方法生成均匀分布向量"""
        self.vae.eval()
        
        with torch.no_grad():
            # 使用球面采样获得更均匀的潜在向量
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            # 归一化到单位球面
            z = F.normalize(z, p=2, dim=1)
            # 随机缩放半径
            radii = torch.rand(num_samples, 1, device=self.device) * 2.5 + 0.5
            z = z * radii
            
            # 通过解码器生成向量
            generated_vectors = self.vae.decode(z)
            
        return generated_vectors.detach()
    
    def dilute_vectors(self, original_vectors, num_new_samples=50):
        """通过生成新的均匀分布向量来稀释原始向量空间"""
        if not self.is_trained:
            raise ValueError("VAE模型尚未训练，请先调用train_vae方法")
            
        # 生成新的均匀分布向量
        new_vectors = self.generate_uniform_vectors(num_samples=num_new_samples)
        
        # 将原始向量和新生成的向量拼接在一起
        diluted_vectors = torch.cat([original_vectors, new_vectors], dim=0)
        
        return diluted_vectors.detach()
        
    def train_vae(self, news_vectors, epochs=1000, lr=1e-3):
        """训练VAE模型"""
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.vae.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.vae(news_vectors)
            loss = self._loss_function(recon_batch, news_vectors, mu, logvar)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f'VAE Training Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
        
        self.is_trained = True
        
    def _loss_function(self, recon_x, x, mu, logvar):
        # 使用MSE损失替代BCE，因为新闻向量不是二进制数据
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
    
    def generate_fake_news(self, user_interest, num_samples=5, num_iterations=500):
        """为特定用户兴趣生成假新闻向量"""
        if not self.is_trained:
            raise ValueError("VAE模型尚未训练，请先调用train_vae方法")
            
        self.vae.eval()
        with torch.no_grad():
            user_mu, user_logvar = self.vae.encode(user_interest)
            
        # 初始化与用户兴趣相反的潜在向量
        anti_latent = torch.nn.Parameter(torch.randn(num_samples, self.latent_dim, device=self.device))
        optimizer = torch.optim.Adam([anti_latent], lr=0.01)
        
        for step in range(num_iterations):
            optimizer.zero_grad()
            generated_news = self.vae.decode(anti_latent)
            
            # 计算与用户兴趣的相似度（我们要最小化这个相似度）
            similarity = F.cosine_similarity(generated_news, user_interest.expand_as(generated_news), dim=1)
            distance = -torch.norm(generated_news - user_interest.expand_as(generated_news), dim=1)
            
            # 组合损失：最小化相似度，最大化距离
            loss = -similarity.mean() + distance.mean() * 0.1
            loss.backward()
            optimizer.step()
            
        return self.vae.decode(anti_latent).detach()