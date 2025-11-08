import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset, get_worker_info
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import numpy as np
from .vae_fake_news_generator import IntegratedFakeNewsGenerator # Import the fake news generator


class TrainDataset(IterableDataset):
    # 训练数据集类
    def __init__(self, filename, news_index, news_input, local_rank, cfg):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = cfg.gpu_num
        # Provide safe defaults for minimal configs (e.g., tests) where his_size may be missing
        try:
            self.his_size = int(cfg.model.his_size)
        except Exception:
            self.his_size = 70

    # 将新闻 ID 转换为新闻索引
    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    # 填充到固定长度
    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    # 数据行映射
    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # 点击新闻处理
        clicked_index, clicked_mask = self.pad_to_fix_len(self.trans_to_nindex(click_id), self.his_size)
        clicked_input = self.news_input[clicked_index]

        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        return clicked_input, clicked_mask, candidate_input, label

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            file_iter = open(self.filename)
            return map(self.line_mapper, file_iter)
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

            def generator():
                with open(self.filename) as f:
                    for i, line in enumerate(f):
                        if i % num_workers != worker_id:
                            continue
                        yield self.line_mapper(line)

            return generator()


class TrainGraphDataset(TrainDataset):
    # 图神经网络训练数据集类，继承自 TrainDataset
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        # Keep dataset tensors on CPU. DataLoader(pin_memory=True) will pin them,
        # and the training loop moves them to GPU with non_blocking copies.
        self.news_graph = news_graph
        # 确保 batch_size 为整数，避免隐式 float 比较导致的边缘开销
        self.batch_size = int(cfg.batch_size / max(1, cfg.gpu_num))
        self.entity_neighbors = entity_neighbors

        # 是否在 DataLoader 中动态生成并注入假新闻（默认关闭，避免重复生成与显存占用）
        self.generate_fake_in_dataloader = getattr(cfg, 'generate_fake_in_dataloader', False)
        if self.generate_fake_in_dataloader:
            # Respect configured device; default to CPU when not CUDA
            if str(getattr(cfg, 'device', 'cpu')).startswith('cuda'):
                target_device = f'cuda:{local_rank}'
            else:
                target_device = str(getattr(cfg, 'device', 'cpu'))
            self.fake_news_generator = IntegratedFakeNewsGenerator(cfg, device=target_device)
            self.fake_news_generator.load_news_encoder(cfg.model.news_encoder_path)
            self.fake_news_generator.initialize_vae()  # 仅在需要时才初始化，避免额外显存占用
        else:
            self.fake_news_generator = None

    def line_mapper(self, line, sum_num_news):
        line = line.strip().split('\t')
        real_click_id = line[3].split()  # 原始历史新闻（预处理阶段已可选地注入过假新闻）

        # 如启用在 DataLoader 内注入，则在此追加；默认关闭以节省显存并避免重复
        if self.generate_fake_in_dataloader and self.fake_news_generator is not None:
            fake_news_ids, _ = self.fake_news_generator.generate_personalized_fake_news_vectors(num_fake_news=20)
            combined_click_id = fake_news_ids + real_click_id
        else:
            combined_click_id = real_click_id
        
        # Handle padding/truncation to his_size (70)
        if len(combined_click_id) > self.his_size:
            click_id = combined_click_id[-self.his_size:] # Truncate from the end if more than 70
        else:
            click_id = combined_click_id # Use as is, padding will be handled by pad_to_fix_len
        
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # ------------------ 点击新闻 ----------------------
        # ------------------ 新闻子图 ---------------------
        top_k = len(click_id)
        click_idx = self.trans_to_nindex(click_id)  # 将点击的新闻 ID 转换为索引
        source_idx = click_idx
        # 根据跳数构建子图
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k, sum_num_news)
        padded_maping_idx = F.pad(mapping_idx, (self.his_size - len(mapping_idx), 0), "constant", -1)

        # ------------------ 候选新闻 ---------------------
        # 训练真实分支需要 [正样本 + 负样本]，并将正样本置于索引0，保证 label=0 有意义
        label = 0
        pos_list = sess_pos  # 单个正样本 id 列表
        neg_list = sess_neg  # npratio 个负样本 id 列表
        sample_ids = pos_list + neg_list
        sample_news = np.atleast_1d(self.trans_to_nindex(sample_ids))
        candidate_input = self.news_input[sample_news]

        # ------------------ 假新闻候选 ---------------------
        fake_sample_news = np.atleast_1d(self.trans_to_nindex(sess_neg))
        fake_candidate_input = self.news_input[fake_sample_news]

        # ------------------ 实体子图 --------------------
        if self.cfg.model.use_entity:
            # 对真实候选（含正+负）构建实体邻居，按 ValidGraphDataset 的模式对齐形状
            origin_entity = candidate_input[:, -3 - self.cfg.model.entity_size:-3]
            cand_cnt = origin_entity.shape[0]
            candidate_neighbor_entity = np.zeros(
                (cand_cnt * self.cfg.model.entity_size, self.cfg.model.entity_neighbors),
                dtype=np.int64)

            # 获取实体邻居
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(
                cand_cnt, self.cfg.model.entity_size * self.cfg.model.entity_neighbors)
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)

            # ------------------ 假新闻实体子图 ---------------------
            fake_origin_entity = fake_candidate_input[:, -3 - self.cfg.model.entity_size:-3]
            fake_candidate_neighbor_entity = np.zeros(
                (self.cfg.npratio * self.cfg.model.entity_size, self.cfg.model.entity_neighbors),
                dtype=np.int64)
            for cnt, idx in enumerate(fake_origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                fake_candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            fake_candidate_neighbor_entity = fake_candidate_neighbor_entity.reshape(self.cfg.npratio,
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)
            fake_entity_mask = fake_candidate_neighbor_entity.copy()
            fake_entity_mask[fake_entity_mask > 0] = 1
            fake_candidate_entity = np.concatenate((fake_origin_entity, fake_candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)
            fake_candidate_entity = np.zeros(1)
            fake_entity_mask = np.zeros(1)

        return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
            fake_candidate_input, fake_candidate_entity, fake_entity_mask, sum_num_news + sub_news_graph.num_nodes

    # 构建新闻子图
    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset:
            subset = [0]

        subset = torch.tensor(subset, dtype=torch.long, device=device)

        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr,
                                                 relabel_nodes=True, num_nodes=self.news_graph.num_nodes)

        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k] + sum_num_nodes

    def __iter__(self):
        worker = get_worker_info()

        def yield_batch(clicked_graphs, candidates, mappings, labels,
                        candidate_entity_list, entity_mask_list,
                        fake_candidate_input_list, fake_candidate_entity_list, fake_entity_mask_list):
            batch = Batch.from_data_list(clicked_graphs)
            candidates = torch.stack(candidates)
            mappings = torch.stack(mappings)
            candidate_entity_list = torch.stack(candidate_entity_list)
            entity_mask_list = torch.stack(entity_mask_list)
            fake_candidate_input_list = torch.stack(fake_candidate_input_list)
            fake_candidate_entity_list = torch.stack(fake_candidate_entity_list)
            fake_entity_mask_list = torch.stack(fake_entity_mask_list)
            labels = torch.tensor(labels, dtype=torch.long)
            return batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels, \
                fake_candidate_input_list, fake_candidate_entity_list, fake_entity_mask_list

        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            fake_candidate_input_list = []
            fake_candidate_entity_list = []
            fake_entity_mask_list = []
            sum_num_news = 0

            with open(self.filename) as f:
                if worker is None:
                    line_iter = enumerate(f)
                else:
                    worker_id = worker.id
                    num_workers = worker.num_workers
                    line_iter = ((i, line) for i, line in enumerate(f) if i % num_workers == worker_id)

                for _, line in line_iter:
                    sub_newsgraph, padded_mapping_idx, candidate_input, candidate_entity, entity_mask, label, \
                        fake_candidate_input, fake_candidate_entity, fake_entity_mask, sum_num_news = self.line_mapper(
                            line, sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)

                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))
                    fake_candidate_input_list.append(torch.from_numpy(fake_candidate_input))
                    fake_candidate_entity_list.append(torch.from_numpy(fake_candidate_entity))
                    fake_entity_mask_list.append(torch.from_numpy(fake_entity_mask))

                    # 当达到 batch_size 时，返回一个 batch
                    if len(clicked_graphs) == self.batch_size:
                        yield yield_batch(clicked_graphs, candidates, mappings, labels,
                                          candidate_entity_list, entity_mask_list,
                                          fake_candidate_input_list, fake_candidate_entity_list, fake_entity_mask_list)

                        clicked_graphs, mappings, candidates, labels = [], [], [], []
                        candidate_entity_list, entity_mask_list = [], []
                        fake_candidate_input_list, fake_candidate_entity_list, fake_entity_mask_list = [], [], []
                        sum_num_news = 0

                # 处理剩余的数据
                if len(clicked_graphs) > 0:
                    yield yield_batch(clicked_graphs, candidates, mappings, labels,
                                      candidate_entity_list, entity_mask_list,
                                      fake_candidate_input_list, fake_candidate_entity_list, fake_entity_mask_list)


class ValidGraphDataset(TrainGraphDataset):
    # 验证数据集类，继承自 TrainGraphDataset
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors,
                 news_entity):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors)
        # Keep on CPU; will be moved to GPU in the validation loop.
        self.news_graph.x = torch.from_numpy(self.news_input)
        self.news_entity = news_entity  # 实体信息

    # 数据行映射处理
    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.his_size:]  # 获取点击的新闻 ID（最近的历史新闻）

        click_idx = self.trans_to_nindex(click_id)  # 将点击的新闻 ID 转换为新闻索引
        clicked_entity = self.news_entity[click_idx]  # 获取点击新闻对应的实体
        source_idx = click_idx

        # 根据跳数构建子图
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])  # 获取当前新闻的邻居节点
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        # 构建新闻子图
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

        # ------------------ 实体 --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])  # 获取标签信息
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])  # 转换候选新闻 ID 为索引
        candidate_input = self.news_input[candidate_index]  # 获取候选新闻输入

        # ------------------ 实体子图 --------------------
        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]  # 获取候选新闻对应的实体
            candidate_neighbor_entity = np.zeros(
                (len(candidate_index) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            # 获取候选新闻的实体邻居
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue  # 如果实体 ID 为 0，则跳过
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue  # 如果该实体没有邻居，则跳过
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)  # 取实体邻居的有效长度
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index),
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)  # 重塑为候选新闻实体邻居矩阵

            entity_mask = candidate_neighbor_entity.copy()  # 实体掩码
            entity_mask[entity_mask > 0] = 1  # 标记有效的实体邻居

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)  # 合并原始实体和邻居实体
        else:
            candidate_entity = np.zeros(1)  # 如果不使用实体，将实体设置为零
            entity_mask = np.zeros(1)  # 同样将实体掩码设置为零

        # 将子图数据转换为 Batch 数据
        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels

    # 数据迭代器
    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:  # 确保当前行有点击新闻信息
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels = self.line_mapper(
                    line)
            yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels


class NewsDataset(Dataset):
    # 新闻数据集类
    def __init__(self, data):
        self.data = data  # 存储数据

    # 获取指定索引的新闻数据
    def __getitem__(self, idx):
        return self.data[idx]

    # 获取数据集的长度
    def __len__(self):
        return self.data.shape[0]  # 返回数据的行数（即样本数）
