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
        return self.fc4(h3)  # 移除sigmoid激活，允许任意范围的输出

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
        # 使用MSE损失替代BCE，因为新闻向量可能不在[0,1]范围内
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
    
    def generate_uniform_space_filling_vectors(self, user_real_vectors, num_fake_per_user=10, 
                                             diversity_weight=1.0, coverage_weight=1.0, 
                                             num_iterations=1000,
                                             proximity_weight=1.0,
                                             stat_weight=0.5,
                                             min_radius=None,
                                             max_radius=None):
        """
        生成空间填充的虚假新闻向量，确保真实和虚假向量均匀覆盖向量空间
        
        Args:
            user_real_vectors: 用户的真实新闻向量 [num_real, vector_dim]
            num_fake_per_user: 每个用户生成的虚假新闻数量
            diversity_weight: 多样性权重，控制虚假向量之间的距离
            coverage_weight: 覆盖权重，控制对整个空间的覆盖
            num_iterations: 优化迭代次数
        
        Returns:
            fake_vectors: 生成的虚假新闻向量
        """
        if not self.is_trained:
            raise ValueError("VAE模型尚未训练，请先调用train_vae方法")
        
        self.vae.eval()
        
        # 1. 分析真实向量的分布特征
        with torch.no_grad():
            real_mu, real_logvar = self.vae.encode(user_real_vectors)
            real_center = real_mu.mean(dim=0)  # 真实向量在潜在空间的中心
            real_std = real_mu.std(dim=0)      # 真实向量在潜在空间的标准差

        # 1.1 估计真实向量在原始空间的典型尺度，用于邻近半径约束
        with torch.no_grad():
            n_real = user_real_vectors.shape[0]
            if n_real >= 2:
                dists = []
                for i in range(n_real):
                    for j in range(i+1, n_real):
                        dists.append(torch.norm(user_real_vectors[i] - user_real_vectors[j]))
                dists = torch.stack(dists)
                typical_radius = torch.median(dists).item()
            else:
                typical_radius = 1.0
        if min_radius is None:
            min_radius = 0.2 * typical_radius
        if max_radius is None:
            max_radius = 1.0 * typical_radius
        
        # 2. 初始化虚假向量的潜在表示
        # 使用更大的方差来确保覆盖更广的空间
        fake_latent = torch.nn.Parameter(
            torch.randn(num_fake_per_user, self.latent_dim, device=self.device) * 2.0
        )
        optimizer = torch.optim.Adam([fake_latent], lr=0.02)
        
        # 3. 优化过程
        for step in range(num_iterations):
            optimizer.zero_grad()
            
            # 解码生成虚假向量
            fake_vectors = self.vae.decode(fake_latent)
            
            # 损失1: 多样性损失 - 确保虚假向量之间有足够距离
            diversity_loss = 0
            if num_fake_per_user > 1:
                for i in range(num_fake_per_user):
                    for j in range(i + 1, num_fake_per_user):
                        # 使用负距离作为损失，鼓励向量分散
                        pairwise_dist = torch.norm(fake_vectors[i] - fake_vectors[j])
                        diversity_loss -= pairwise_dist
                diversity_loss /= (num_fake_per_user * (num_fake_per_user - 1) / 2)
            
            # 损失2: 覆盖损失（保留原有覆盖推动，但由权重控制强度）
            coverage_loss = 0
            for fake_vec in fake_vectors:
                min_dist_to_real = torch.min(torch.norm(fake_vec.unsqueeze(0) - user_real_vectors, dim=1))
                coverage_loss -= min_dist_to_real
            coverage_loss /= num_fake_per_user

            # 损失3: 邻近半径约束 - 使虚假向量既不贴得太近也不离得太远
            proximity_loss = 0
            for fake_vec in fake_vectors:
                min_dist_to_real = torch.min(torch.norm(fake_vec.unsqueeze(0) - user_real_vectors, dim=1))
                over_pen = torch.relu(min_dist_to_real - max_radius)
                under_pen = torch.relu(min_radius - min_dist_to_real)
                proximity_loss += (over_pen + under_pen)
            proximity_loss /= num_fake_per_user

            # 损失4: 潜在空间均匀分布损失
            latent_uniform_loss = 0
            # 鼓励潜在向量在潜在空间中均匀分布
            latent_center = fake_latent.mean(dim=0)
            # 惩罚潜在向量过于集中
            latent_concentration = torch.norm(fake_latent - latent_center.unsqueeze(0), dim=1).var()
            latent_uniform_loss = -latent_concentration  # 最大化方差

            # 损失5: 潜在分布匹配损失 - 使假样本的潜在均值与方差贴近真实分布
            fake_mu = fake_latent.mean(dim=0)
            fake_std = fake_latent.std(dim=0) + 1e-6
            stat_match_loss = F.mse_loss(fake_mu, real_center) + F.mse_loss(fake_std, real_std)
                        
            # 组合所有损失
            total_loss = (diversity_weight * diversity_loss + 
                         coverage_weight * coverage_loss + 
                         proximity_weight * proximity_loss +
                         0.5 * latent_uniform_loss +
                         stat_weight * stat_match_loss)
            
            total_loss.backward()
            optimizer.step()
            
            # 定期打印进度
            if step % 200 == 0:
                print(f"Step {step}: Total Loss: {total_loss.item():.4f}, "
                      f"Diversity: {diversity_loss.item():.4f}, "
                      f"Coverage: {coverage_loss.item():.4f}")
        
        # 返回生成的虚假向量
        with torch.no_grad():
            final_fake_vectors = self.vae.decode(fake_latent)
        
        return final_fake_vectors.detach()

    def generate_pca_density_matched_vectors(self, all_real_vectors, num_fake_total, pca_components=50,
                                             k=10, spread=0.7, jitter=0.05, random_state=42):
        """
        在PCA空间进行密度匹配采样：基于真实数据的聚类中心和每维方差进行高斯混合采样，
        生成贴近真实分布但具有多样性的假向量，避免与真实向量明显可分。

        Args:
            all_real_vectors: 所有真实新闻向量 [num_all_real, dim]
            num_fake_total: 要生成的虚假向量总数
            pca_components: PCA降维维度
            k: KMeans聚类簇数
            spread: 采样标准差缩放系数（越大越分散）
            jitter: 在原始空间加入的小抖动幅度
            random_state: 随机种子

        Returns:
            fake_vectors: 贴近真实分布的虚假向量（torch.Tensor）
        """
        if not self.is_trained:
            raise ValueError("VAE模型尚未训练，请先调用train_vae方法")

        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        real_np = all_real_vectors.cpu().numpy()
        pca = PCA(n_components=pca_components, random_state=random_state)
        real_pca = pca.fit_transform(real_np)

        k = int(max(1, min(k, len(real_pca))))
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = kmeans.fit_predict(real_pca)
        centers = kmeans.cluster_centers_

        # 计算每个簇的每维方差，作为高斯采样的尺度
        cluster_vars = np.zeros((k, pca_components), dtype=np.float32)
        for j in range(k):
            pts = real_pca[labels == j]
            if len(pts) >= 2:
                cluster_vars[j] = np.var(pts, axis=0)
            else:
                cluster_vars[j] = np.var(real_pca, axis=0) * 0.5

        cluster_sizes = np.bincount(labels, minlength=k).astype(np.float64)
        probs = cluster_sizes / cluster_sizes.sum() if cluster_sizes.sum() > 0 else np.ones(k) / k

        rng = np.random.default_rng(random_state)
        samples = []
        for i in range(num_fake_total):
            j = rng.choice(k, p=probs)
            mean = centers[j]
            std = np.sqrt(cluster_vars[j] + 1e-6) * spread
            sample = rng.normal(loc=mean, scale=std)
            samples.append(sample)

        fake_pca = np.vstack(samples)
        fake_original = pca.inverse_transform(fake_pca)
        fake_vectors_torch = torch.tensor(fake_original, dtype=torch.float32, device=self.device)
        if jitter and jitter > 0:
            fake_vectors_torch = fake_vectors_torch + torch.randn_like(fake_vectors_torch) * jitter

        return fake_vectors_torch
    
    def generate_pca_uniform_vectors(self, all_real_vectors, num_fake_total, pca_components=50):
        """
        基于PCA分析生成在降维空间中均匀分布的虚假向量
        
        Args:
            all_real_vectors: 所有真实新闻向量 [num_all_real, vector_dim]
            num_fake_total: 总共要生成的虚假向量数量
            pca_components: PCA降维的维度数
        
        Returns:
            fake_vectors: 在PCA空间中均匀分布的虚假向量
        """
        if not self.is_trained:
            raise ValueError("VAE模型尚未训练，请先调用train_vae方法")
        
        # 1. 对真实向量进行PCA分析
        from sklearn.decomposition import PCA
        
        real_np = all_real_vectors.cpu().numpy()
        pca = PCA(n_components=pca_components)
        real_pca = pca.fit_transform(real_np)
        
        # 2. 分析PCA空间的边界
        pca_min = real_pca.min(axis=0)
        pca_max = real_pca.max(axis=0)
        pca_range = pca_max - pca_min
        
        # 3. 在PCA空间中生成均匀分布的点
        # 扩展边界以确保覆盖更大空间
        expanded_min = pca_min - 0.3 * pca_range
        expanded_max = pca_max + 0.3 * pca_range
        
        # 使用拉丁超立方采样确保均匀分布
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=pca_components)
        uniform_samples = sampler.random(n=num_fake_total)
        
        # 缩放到PCA空间范围
        fake_pca = expanded_min + uniform_samples * (expanded_max - expanded_min)
        
        # 4. 将PCA空间的点转换回原始空间
        fake_original = pca.inverse_transform(fake_pca)
        fake_vectors_torch = torch.tensor(fake_original, dtype=torch.float32, device=self.device)
        
        # 5. 使用VAE的解码器进一步优化这些向量
        # 将原始空间向量编码到潜在空间，然后解码
        with torch.no_grad():
            # 这里我们直接使用生成的向量，不通过VAE编码-解码
            # 因为我们希望保持PCA空间的均匀分布特性
            pass
        
        return fake_vectors_torch