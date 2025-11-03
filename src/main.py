import os.path
from pathlib import Path

import hydra
import logging

print("[DEBUG] main.py started.")
#import wandb
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda import amp
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from dataload.data_load import load_data
from dataload.data_preprocess import prepare_preprocessed_data
from utils.metrics import *
from utils.common import *

# Prefer faster matmul kernels when available (PyTorch >= 2.0)
try:
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        # Allow TF32 on Ampere+ for extra throughput
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True        # type: ignore[attr-defined]
except Exception:
    pass

### custom your wandb setting here ###
#os.environ["WANDB_API_KEY"] = "f01277e9a4a605e59d5c667b58ac365388f90fa0"
#os.environ["WANDB_MODE"] = "online"  # 设置wandb运行模式为离线
# 让用户自行通过环境变量选择可见 GPU，避免强制占用 GPU:0 导致 OOM
os.environ["WANDB_DISABLE"] = "true"

# ------------------------- Simple logger setup
# Use a lightweight console logger since wandb is disabled.
# Keep it process-safe: each spawned process configures its own logger.
logger = logging.getLogger("glory")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def _select_device_for_rank(local_rank: int):
    """Select best available device for the given rank.
    - cuda: use cuda:<local_rank>
    - mps: Apple Silicon accelerator
    - cpu: fallback
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def train(model, optimizer, scaler, scheduler, dataloader, local_rank, device, cfg, early_stopping, is_distributed=False):
    model.train()  # 设置模型为训练模式
    torch.set_grad_enabled(True)  # 开启梯度计算

    sum_loss = torch.zeros(1, device=device)  # 存储累积损失
    sum_auc = torch.zeros(1, device=device)  # 存储累积AUC（Area Under Curve）
    # 添加假新闻相关指标
    sum_70_news_auc = torch.zeros(1, device=device)  # 存储基于 70 条新闻的 AUC
    
    # 遍历数据集进行训练
    for cnt, (subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels, \
            fake_candidate_news, fake_candidate_entity, fake_entity_mask) \
            in enumerate(tqdm(dataloader,
                              total=int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)),
                              desc=f"[{local_rank}] Training"), start=1):

        # 将数据移到计算设备上
        subgraph = subgraph.to(device, non_blocking=True)
        mapping_idx = mapping_idx.to(device, non_blocking=True)
        candidate_news = candidate_news.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        candidate_entity = candidate_entity.to(device, non_blocking=True)
        entity_mask = entity_mask.to(device, non_blocking=True)
        fake_candidate_news = fake_candidate_news.to(device, non_blocking=True)
        fake_candidate_entity = fake_candidate_entity.to(device, non_blocking=True)
        fake_entity_mask = fake_entity_mask.to(device, non_blocking=True)

        # 使用自动混合精度训练（仅 CUDA 启用）
        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            loss, score_real, score_70_news = model(subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask,
                                                    fake_candidate_news, fake_candidate_entity, fake_entity_mask, labels)

        # 累积梯度
        scaler.scale(loss).backward()

        # 梯度更新周期
        if cnt % cfg.accumulation_steps == 0 or cnt == int(cfg.dataset.pos_count / cfg.batch_size):
            # 更新参数
            scaler.step(optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()  # 更新scaler的值
            new_scaler = scaler.get_scale()
            if new_scaler >= old_scaler:
                scheduler.step()  # 调整学习率
                ## https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814
            optimizer.zero_grad(set_to_none=True)  # 清除梯度

        sum_loss += loss.data.float()  # 累加损失
        sum_auc += area_under_curve(labels, score_real)  # 累加AUC指标
    
        sum_70_news_auc += area_under_curve(labels, score_70_news)  # 累加基于 70 条新闻的 AUC

        # ---------------------------------------- 训练日志
        if cnt % cfg.log_steps == 0:
            logger.info('[{}]: Ed: {}, average_loss: {:.5f}, real_news_auc: {:.5f}, 70_news_auc: {:.5f}'.format(
                local_rank, cnt * cfg.batch_size, 
                sum_loss.item() / cfg.log_steps, 
                sum_auc.item() / cfg.log_steps,
                sum_70_news_auc.item() / cfg.log_steps))
            sum_loss.zero_()  # 清零累积损失
            sum_auc.zero_()  # 清零累积AUC
            sum_70_news_auc.zero_()  # 清零基于 70 条新闻的 AUC

        # 如果超过一定步数，进行验证
        if cnt > int(cfg.val_skip_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)) and cnt % cfg.val_steps == 0:
            res = val(model, device, cfg, local_rank=local_rank, is_distributed=is_distributed)
            model.train()  # 验证后切换回训练模式

            if local_rank == 0:
                logger.info(
                    "Validation | AUC_real: {:.4f} | MRR_real: {:.4f} | NDCG5_real: {:.4f} | NDCG10_real: {:.4f} | AUC_70: {:.4f} | MRR_70: {:.4f} | NDCG5_70: {:.4f} | NDCG10_70: {:.4f}".format(
                        res["auc_real"], res["mrr_real"], res["ndcg5_real"], res["ndcg10_real"], res["auc_70"], res["mrr_70"], res["ndcg5_70"], res["ndcg10_70"]
                    )
                )
                #wandb.log(res)  # 记录到wandb

            early_stop, get_better = early_stopping(res['auc_real'])  # 判断是否提前停止
            if early_stop:
                print("Early Stop.")
                break  # 提前停止
            elif get_better:
                print(f"Better Result!")
                if local_rank == 0:
                    # 保存模型并记录最好的AUC
                    save_model(cfg, model, optimizer, f"{cfg.ml_label}_auc{res['auc_real']}")
                    #wandb.run.summary.update({"best_auc": res["auc"], "best_mrr": res['mrr'],
                                             # "best_ndcg5": res['ndcg5'], "best_ndcg10": res['ndcg10']})


def val(model, device, cfg, local_rank=0, is_distributed=False):
    model.eval()  # 设置模型为评估模式
    dataloader = load_data(cfg, mode='val', model=model, local_rank=local_rank, device=device)  # 加载验证数据
    tasks = []
    # 评估阶段使用 inference_mode 进一步减少开销
    with torch.inference_mode():  # 评估时不计算梯度且更省显存
        for cnt, (subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, labels) \
                in enumerate(tqdm(dataloader,
                                  total=int(cfg.dataset.val_len / max(1, cfg.gpu_num)),
                                  desc=f"[{local_rank}] Validating")):
            # 将数据移到计算设备上
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(device, non_blocking=True)
            candidate_entity = candidate_entity.to(device, non_blocking=True)
            entity_mask = entity_mask.to(device, non_blocking=True)
            clicked_entity = clicked_entity.to(device, non_blocking=True)

            # 模型评估阶段的处理过程
            module = getattr(model, 'module', model)
            scores_real, scores_70_news = module.validation_process(subgraph, mappings, clicked_entity, candidate_emb,
                                                     candidate_entity, entity_mask)

            tasks.append((labels.cpu().numpy(), scores_real.cpu().numpy(), scores_70_news.cpu().numpy()))

    # 使用多进程池计算各项指标
    with mp.Pool(processes=cfg.num_workers) as pool:
        results = pool.map(cal_metric, tasks)  # 计算每个任务的指标
    val_auc_real, val_mrr_real, val_ndcg5_real, val_ndcg10_real, val_auc_70, val_mrr_70, val_ndcg5_70, val_ndcg10_70 = np.array(results).T  # 提取各个指标

    # barrier
    if is_distributed and dist.is_available() and dist.is_initialized():
        torch.distributed.barrier()  # 用于同步不同GPU的进程

    # 汇总不同GPU的结果
    reduced_auc_real = reduce_mean(torch.tensor(np.nanmean(val_auc_real)).float().to(device), cfg.gpu_num)
    reduced_mrr_real = reduce_mean(torch.tensor(np.nanmean(val_mrr_real)).float().to(device), cfg.gpu_num)
    reduced_ndcg5_real = reduce_mean(torch.tensor(np.nanmean(val_ndcg5_real)).float().to(device), cfg.gpu_num)
    reduced_ndcg10_real = reduce_mean(torch.tensor(np.nanmean(val_ndcg10_real)).float().to(device), cfg.gpu_num)

    reduced_auc_70 = reduce_mean(torch.tensor(np.nanmean(val_auc_70)).float().to(device), cfg.gpu_num)
    reduced_mrr_70 = reduce_mean(torch.tensor(np.nanmean(val_mrr_70)).float().to(device), cfg.gpu_num)
    reduced_ndcg5_70 = reduce_mean(torch.tensor(np.nanmean(val_ndcg5_70)).float().to(device), cfg.gpu_num)
    reduced_ndcg10_70 = reduce_mean(torch.tensor(np.nanmean(val_ndcg10_70)).float().to(device), cfg.gpu_num)

    res = {
        "auc_real": reduced_auc_real.item(),
        "mrr_real": reduced_mrr_real.item(),
        "ndcg5_real": reduced_ndcg5_real.item(),
        "ndcg10_real": reduced_ndcg10_real.item(),
        "auc_70": reduced_auc_70.item(),
        "mrr_70": reduced_mrr_70.item(),
        "ndcg5_70": reduced_ndcg5_70.item(),
        "ndcg10_70": reduced_ndcg10_70.item(),
    }

    return res


def main_worker(local_rank, cfg):
    # -----------------------------------------环境初始化
    seed_everything(cfg.seed)  # 设置随机种子
    device = _select_device_for_rank(local_rank)
    is_distributed = (device.type == 'cuda' and cfg.gpu_num > 1)
    if is_distributed:
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:23456',
                                world_size=cfg.gpu_num,
                                rank=local_rank)

    # -----------------------------------------加载数据集和模型
    num_training_steps = int(cfg.num_epochs * cfg.dataset.pos_count / (cfg.batch_size * cfg.accumulation_steps))
    #accumulation_steps内存问题，用小batch——size,但是每个batch之后不更新，而是累积几个step一起更新，从而模拟更大批次的训练
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio + 1)  # 计算warmup步数
    train_dataloader = load_data(cfg, mode='train', local_rank=local_rank, device=device)  # 加载训练数据
    model = load_model(cfg).to(device)  # 加载模型并移到对应设备
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)  # 初始化优化器

    lr_lambda = lambda step: 1.0 if step > num_warmup_steps else step / num_warmup_steps
    scheduler = LambdaLR(optimizer, lr_lambda)  # 学习率调整

    # ------------------------------------------加载检查点（如果需要）
    if cfg.load_checkpoint:
        file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{cfg.load_mark}.pth")
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  # 分布式训练
    optimizer.zero_grad(set_to_none=True)  # 清除优化器的梯度
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))  # 混合精度仅在 CUDA 启用

    # ------------------------------------------训练开始
    early_stopping = EarlyStopping(cfg.early_stop_patience)  # 提前停止

    if local_rank == 0:
        #wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
        #           project=cfg.logger.exp_name, name=cfg.logger.run_name)  # 初始化wandb
        print(model)

    # 启动训练
    train(model, optimizer, scaler, scheduler, train_dataloader, local_rank, device, cfg, early_stopping, is_distributed=is_distributed)

    #if local_rank == 0:
        #wandb.finish()  # 结束wandb记录


@hydra.main(version_base="1.2", config_path=os.path.join(get_root(), "configs"), config_name="small")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)  # 设置随机种子
    # 选择设备与并行策略：非 CUDA 环境下回退为单进程
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        cfg.gpu_num = torch.cuda.device_count()
    else:
        cfg.gpu_num = 1
    prepare_preprocessed_data(cfg)  # 准备预处理数据
    if cfg.gpu_num > 1:
        mp.spawn(main_worker, nprocs=cfg.gpu_num, args=(cfg,))  # 启动多进程训练
    else:
        main_worker(0, cfg)  # 单进程训练（CPU/MPS/单卡）


if __name__ == "__main__":
    main()  # 启动主函数
