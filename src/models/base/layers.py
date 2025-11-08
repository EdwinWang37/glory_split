import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.base.function import *


class DotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, left, right):
        """

        Args:
            left: (batch_size, num_candidate, dim_embedding)
            right: (batch_size, dim_embedding)

        Returns:
            (shape): batch_size, candidate_num

        """
        result = torch.bmm(left, right.unsqueeze(dim=-1)).squeeze(dim=-1)
        return result


class AttentionPooling(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def initialize(self):
        nn.init.xavier_uniform_(self.att_fc1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.att_fc1.bias)
        nn.init.xavier_uniform_(self.att_fc2.weight)
       
    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)  # [B, L, 1]

        # 数值稳定性更好的 masked softmax
        if attn_mask is not None:
            alpha = alpha.masked_fill(attn_mask.unsqueeze(2) == 0, float('-inf'))
        alpha = torch.softmax(alpha, dim=1)
        # 如果整行都被 mask，softmax(-inf) 会产生 NaN；将其安全置零
        if attn_mask is not None:
            all_masked = (attn_mask.sum(dim=1) == 0).unsqueeze(1).unsqueeze(2)
            alpha = torch.where(all_masked, torch.zeros_like(alpha), alpha)
        # 兜底清理可能的 NaN/Inf
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))

        x = torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        """
            Q: [B, H, Lq, d_k]
            K: [B, H, Lk, d_k]
            V: [B, H, Lk, d_v]
            attn_mask: [B, H, Lk]
            Return: [B, H, Lq, d_v]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(-2) == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        # 当整条序列都被 mask 时，softmax(-inf) -> NaN；为这些条目置零
        if attn_mask is not None:
            # attn_mask: [B, H, Lk]
            # attn:      [B, H, Lq, Lk]
            # Broadcast condition to both Lq and Lk dimensions
            all_masked = (attn_mask.sum(dim=-1) == 0).unsqueeze(-1).unsqueeze(-1)
            attn = torch.where(all_masked, torch.zeros_like(attn), attn)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, head_num, head_dim, residual=False):
        super().__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.residual = residual

        self.W_Q = nn.Linear(key_size, self.head_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(query_size, self.head_dim * self.head_num, bias=False)
        self.W_V = nn.Linear(value_size, self.head_dim * self.head_num, bias=True)

        self.scaled_dot_product_attn = ScaledDotProductAttention(self.head_dim)
        self.apply(xavier)

    def initialize(self):
        nn.init.zeros_(self.W_Q.bias)
        nn.init.zeros_(self.W_V.bias)


    def forward(self, Q, K, V, mask=None):
        """
            Q: batch_size, candidate_num, news_dim
            K: batch_size, candidate_num, news_dim
            V: batch_size, candidate_num, news_dim
            mask: batch_size, candidate_num
        """
        batch_size = Q.shape[0]
        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.head_num, -1)

        q_s = self.W_Q(Q).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)

        context = self.scaled_dot_product_attn(q_s, k_s, v_s, mask)
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.head_dim)
        if self.residual:
            output += Q
        return output
