import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GatedGraphConv

from models.base.layers import *
from models.component.candidate_encoder import *
from models.component.click_encoder import ClickEncoder
from models.component.entity_encoder import EntityEncoder, GlobalEntityEncoder
from models.component.nce_loss import NCELoss
from models.component.news_encoder import *
from models.component.user_encoder import *
from models.component.noise_encoder import *


class GLORYServer(nn.Module):
    '''
    服务器只计算global gnn
    '''
    def __init__(self, cfg, glove_emb=None,):
        super().__init__()
        self.cfg = cfg
        self.news_dim = cfg.model.head_num * cfg.model.head_dim  # 新闻的维度

        # GCN（图卷积网络）
        self.global_news_encoder = Sequential('x, index', [
            (GatedGraphConv(self.news_dim, num_layers=3, aggr='add'), 'x, index -> x'),  # 采用 GatedGraphConv 进行图卷积
        ])

    def forward(self, x_encoded, edge_index, mapping_idx):
        # 服务器端计算图卷积特征
        graph_emb = self.global_news_encoder(x_encoded, edge_index)
        # clicked_graph_emb = graph_emb[mapping_idx, :]

        return graph_emb


class GLORYClient(nn.Module):
    '''
    客户端负责本地新闻的语义计算和后续全局图向量传入的计算
    '''
    def __init__(self, cfg, entity_emb, glove_emb=None, ):
        super().__init__()
        self.cfg = cfg
        self.news_dim = cfg.model.head_num * cfg.model.head_dim  # 新闻的维度
        self.local_news_encoder = NewsEncoder(cfg, glove_emb)
        self.click_encoder = ClickEncoder(cfg)
        self.user_encoder = UserEncoder(cfg)
        self.candidate_encoder = CandidateEncoder(cfg)
        self.click_predictor = DotProduct()
        self.use_entity = cfg.model.use_entity
        self.entity_dim = cfg.model.entity_emb_dim  # 实体的维度
        self.loss_fn = NCELoss()
        # 实体编码器
        if self.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()  # 使用预训练的实体嵌入
            self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)  # 实体嵌入层

            # 局部实体编码器
            self.local_entity_encoder = Sequential('x, mask', [
                (self.entity_embedding_layer, 'x -> x'),
                (EntityEncoder(cfg), 'x, mask -> x'),  # 使用 EntityEncoder 进行实体编码
            ])
            self.global_entity_encoder = Sequential('x, mask', [
                (self.entity_embedding_layer, 'x -> x'),
                (GlobalEntityEncoder(cfg), 'x, mask -> x'),  # 使用 GlobalEntityEncoder 进行全局实体编码
            ])


    def combine_embeddings(self, clicked_origin_emb, clicked_graph_emb, clicked_entity, mask, batch_size, num_clicked):
        # 合并原始特征和图卷积特征
        clicked_total_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity)
        user_emb = self.user_encoder(clicked_total_emb, mask)  # 用户嵌入
        return user_emb

    def predict_click(self, user_emb, candidate_news, candidate_entity, entity_mask):
        # 计算点击预测分数
        # --------------------------------------------1--------------------------------------------
        # ----------------------------------------- 处理候选新闻 ------------------------------------
        # 对候选新闻进行编码
        # 支持两种输入：
        # 1) 训练时传入原始 token（[B, N, total_input]）-> 需要编码
        # 2) 验证时传入已编码的向量（[N, news_dim] 或 [1, N, news_dim]）-> 直接使用
        if candidate_news.dim() == 2 and candidate_news.shape[-1] == self.news_dim and torch.is_floating_point(candidate_news):
            cand_title_emb = candidate_news.unsqueeze(0)
        elif candidate_news.dim() == 3 and candidate_news.shape[-1] == self.news_dim and torch.is_floating_point(candidate_news):
            cand_title_emb = candidate_news
        else:
            cand_title_emb = self.local_news_encoder(candidate_news)  # 编码候选新闻标题
        if self.use_entity:
            # 分离候选新闻的原始实体和邻居实体
            origin_entity, neighbor_entity = candidate_entity.split(
                [self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            # 对候选新闻的实体进行编码
            # 兼容验证阶段 [N, ...] 的形状，补齐 batch 维度
            if origin_entity.dim() == 2:
                origin_entity = origin_entity.unsqueeze(0)
            if neighbor_entity.dim() == 2:
                neighbor_entity = neighbor_entity.unsqueeze(0)
            if entity_mask is not None and entity_mask.dim() == 2:
                entity_mask = entity_mask.unsqueeze(0)

            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)

        else:
            cand_origin_entity_emb, cand_neighbor_entity_emb = None, None

        # 将候选新闻的嵌入信息通过候选新闻编码器合并
        cand_final_emb = self.candidate_encoder(cand_title_emb, cand_origin_entity_emb, cand_neighbor_entity_emb)
        score = self.click_predictor(cand_final_emb, user_emb)
        return score

    def process_news(self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, clicked_entity_input=None):
        # 处理新闻的编码过程
        # 兼容 DataLoader 在验证阶段返回的一维 mapping_idx
        if mapping_idx.dim() == 1:
            mapping_idx = mapping_idx.unsqueeze(0)
        mask = mapping_idx != -1
        mapping_idx = mapping_idx.clone()  # 避免原地修改上游变量
        mapping_idx[mapping_idx == -1] = 0
        batch_size, num_clicked = mapping_idx.shape[0], mapping_idx.shape[1]
        token_dim = candidate_news.shape[-1]

        # 如果 subgraph.x 已经是编码后的向量（验证阶段），则跳过本地编码器
        x_feat = subgraph.x
        if torch.is_floating_point(x_feat) and x_feat.shape[-1] == self.news_dim:
            x_encoded = x_feat
            if self.use_entity:
                # 验证阶段使用数据集中提供的实体索引
                if clicked_entity_input is not None:
                    ce = clicked_entity_input
                    if ce.dim() == 2:
                        ce = ce.unsqueeze(0)
                    clicked_entity = self.local_entity_encoder(ce, None)
                else:
                    clicked_entity = None
            else:
                clicked_entity = None
            return x_encoded, mask, mapping_idx, batch_size, num_clicked, clicked_entity

        # 训练阶段：subgraph.x 仍为原始 token，需要编码
        clicked_entity = subgraph.x[mapping_idx, -8:-3]  # 获取点击新闻的实体信息
        x_flatten = subgraph.x.view(1, -1, token_dim)  # 展平
        x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)  # 编码
        if self.use_entity:
            clicked_entity = self.local_entity_encoder(clicked_entity, None)  # 对点击新闻的实体进行编码
        else:
            clicked_entity = None
        return x_encoded, mask, mapping_idx, batch_size, num_clicked, clicked_entity


class GLORYSplit(nn.Module):
    '''
    完整的模型
    '''
    def __init__(self, cfg, glove_emb=None, entity_emb=None):
        super().__init__()
        self.cfg = cfg
        self.client = GLORYClient(cfg, entity_emb, glove_emb,)
        self.server = GLORYServer(cfg, glove_emb)
        self.use_entity = cfg.model.use_entity
        #self.noise = NoiseAddition(0.1)
        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        #self.denoise = DenoisingMLP(self.news_dim, 1024)

        # 添加假新闻生成器（如果需要动态生成）
        # self.fake_news_generator = FakeNewsGenerator(cfg)

    def compute_security_loss(self, score_real, score_fake_candidate, label):
        # 这是一个占位符实现，需要根据实际的损失函数逻辑进行修改
        # 例如，可以计算交叉熵损失或自定义损失
        # 这里简单返回一个常数作为示例
        if label is not None:
            # 假设我们有一个简单的二分类交叉熵损失
            loss_real = self.client.loss_fn(score_real, label)
            loss_fake = self.client.loss_fn(score_fake_candidate, torch.zeros_like(label)) # 假设假新闻的标签是0
            return loss_real + loss_fake
        return torch.tensor(0.0)

    def validation_process(self, subgraph, mapping_idx, clicked_entity, candidate_news, candidate_entity, entity_mask):
        # 客户端计算 x_encoded 和其他特征 (现在用于 70 条历史新闻)
        x_encoded, mask, mapping_idx, batch_size, num_clicked, clicked_entity = self.client.process_news(
            subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, clicked_entity_input=clicked_entity)

        # Ensure tensors are on the same device as the model before GNN and masking
        dev = next(self.parameters()).device
        x_encoded = x_encoded.to(dev)
        mapping_idx = mapping_idx.to(dev)
        mask = mask.to(dev)
        if clicked_entity is not None:
            clicked_entity = clicked_entity.to(dev)

        clicked_origin_emb_70 = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                                self.client.news_dim)
        graph_emb = self.server(x_encoded, subgraph.edge_index.to(dev), mapping_idx)

        clicked_graph_emb_70 = graph_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                                self.client.news_dim)

        user_emb_70 = self.client.combine_embeddings(clicked_origin_emb_70, clicked_graph_emb_70, clicked_entity, mask,
                                                  batch_size, num_clicked)

        # 1. 计算基于 50 条真实新闻的推荐分数 (y_hat)
        num_real_news = 50
        clicked_origin_emb_real = clicked_origin_emb_70[:, -num_real_news:, :]
        clicked_graph_emb_real = clicked_graph_emb_70[:, -num_real_news:, :]
        clicked_entity_real = clicked_entity[:, -num_real_news:, :]
        mask_real = mask[:, -num_real_news:]

        user_emb_real = self.client.combine_embeddings(clicked_origin_emb_real, clicked_graph_emb_real, clicked_entity_real, mask_real,
                                                  batch_size, num_real_news)
        score_real = self.client.predict_click(user_emb_real, candidate_news, candidate_entity, entity_mask)

        # 2. 计算基于 70 条新闻（真实 + 虚假）的推荐分数 (y_hat_fake in main.py)
        score_70_news = self.client.predict_click(user_emb_70, candidate_news, candidate_entity, entity_mask)

        return score_real, score_70_news

    def forward(self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask,
                fake_candidate_news, fake_candidate_entity, fake_entity_mask, label=None):
        """
        Args:
            subgraph: 新闻子图
            mapping_idx: 点击新闻映射索引
            candidate_news: 真实候选新闻
            candidate_entity: 真实候选新闻实体
            entity_mask: 真实新闻实体掩码
            fake_candidate_news: 假新闻候选
            fake_candidate_entity: 假新闻实体
            fake_entity_mask: 假新闻实体掩码
            label: 真实新闻标签
        """
        # 客户端计算 x_encoded 和其他特征 (现在用于 70 条历史新闻)
        x_encoded, mask, mapping_idx, batch_size, num_clicked, clicked_entity = self.client.process_news(
            subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask) # num_clicked 将是 70

        # Align devices for safety when running on different executors
        dev = next(self.parameters()).device
        x_encoded = x_encoded.to(dev)
        mapping_idx = mapping_idx.to(dev)
        mask = mask.to(dev)
        if clicked_entity is not None:
            clicked_entity = clicked_entity.to(dev)

        clicked_origin_emb_70 = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                                self.client.news_dim)
        # 加噪声
        #x_encoded = self.denoise(x_encoded)
        # 服务器端计算 clicked_graph_emb
        graph_emb = self.server(x_encoded, subgraph.edge_index.to(dev), mapping_idx)
        # 去噪声
        #graph_emb = self.denoise(graph_emb)

        clicked_graph_emb_70 = graph_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                                self.client.news_dim)

        user_emb_70 = self.client.combine_embeddings(clicked_origin_emb_70, clicked_graph_emb_70, clicked_entity, mask,
                                                  batch_size, num_clicked)

        # 1. 计算基于 50 条真实新闻的推荐分数 (y_hat)
        num_real_news = 50
        clicked_origin_emb_real = clicked_origin_emb_70[:, -num_real_news:, :]
        clicked_graph_emb_real = clicked_graph_emb_70[:, -num_real_news:, :]
        clicked_entity_real = clicked_entity[:, -num_real_news:, :]
        mask_real = mask[:, -num_real_news:]

        user_emb_real = self.client.combine_embeddings(clicked_origin_emb_real, clicked_graph_emb_real, clicked_entity_real, mask_real,
                                                  batch_size, num_real_news)
        score_real = self.client.predict_click(user_emb_real, candidate_news, candidate_entity, entity_mask)

        # 2. 计算基于 70 条新闻（真实 + 虚假）的推荐分数 (y_hat_fake in main.py)
        score_70_news = self.client.predict_click(user_emb_70, candidate_news, candidate_entity, entity_mask)

        # 3. 计算假新闻候选的推荐分数 (用于损失计算)
        score_fake_candidate = self.client.predict_click(user_emb_70, fake_candidate_news, fake_candidate_entity, fake_entity_mask)

        # 4. 计算安全损失函数
        loss = self.compute_security_loss(score_real, score_fake_candidate, label)

        return loss, score_real, score_70_news
