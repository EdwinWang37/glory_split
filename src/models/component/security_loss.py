import torch
import torch.nn as nn
import torch.nn.functional as F

class SecurityNCELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha  # 真实新闻损失权重
        self.beta = beta    # 假新闻对抗损失权重
    
    def forward(self, score_real, score_fake, label_real):
        """
        Args:
            score_real: 真实新闻分数 (batch_size, real_candidate_num)
            score_fake: 假新闻分数 (batch_size, fake_candidate_num)
            label_real: 真实新闻标签 (batch_size,)
        """
        # 1. 真实新闻NCE损失
        real_log_prob = F.log_softmax(score_real, dim=1)
        loss_real = F.nll_loss(real_log_prob, label_real)
        
        # 2. 假新闻对抗损失 - 最小化假新闻的推荐概率
        fake_log_prob = F.log_softmax(score_fake, dim=1)
        loss_fake = -fake_log_prob.mean()  # 负对数似然，使假新闻概率降低
        
        # 3. 组合损失
        total_loss = self.alpha * loss_real + self.beta * loss_fake
        
        return total_loss