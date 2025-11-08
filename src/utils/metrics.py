import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F


def area_under_curve(y_true, y_hat):
    """
    Function for calculating the auc.
    Inputs:
        y_true - True labels
        y_hat - Predicted labels
    Outputs:
        auc - auc between the predicted and real labels
    """

    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)


def cal_metric(args):
    """Compute metrics for one impression with robust NaN handling.

    - Filters out any pairs where either label or score is non-finite.
    - Returns AUC as NaN if only one class remains (so the caller's nanmean can ignore it).
    - Returns 0.0 for MRR/NDCG when there are no positive labels.
    """
    labels, scores_real, scores_70_news = args

    def _sanitize(y_true, y_score):
        yt = np.asarray(y_true).reshape(-1).astype(float)
        ys = np.asarray(y_score).reshape(-1).astype(float)
        mask = np.isfinite(yt) & np.isfinite(ys)
        yt = yt[mask]
        ys = ys[mask]
        return yt, ys

    def _safe_auc(yt, ys):
        # Need both positive and negative labels to compute ROC-AUC
        if yt.size == 0 or np.unique(yt).size < 2:
            return np.nan
        try:
            return roc_auc_score(yt, ys)
        except Exception:
            return np.nan

    def _safe_mrr(yt, ys):
        if yt.size == 0 or np.sum(yt) == 0:
            return 0.0
        return mrr_score(yt, ys)

    def _safe_ndcg(yt, ys, k):
        if yt.size == 0 or np.sum(yt) == 0:
            return 0.0
        # Guard divide-by-zero inside ndcg when ideal DCG is 0
        ideal = dcg_score(yt, yt, k)
        if ideal == 0:
            return 0.0
        return ndcg_score(yt, ys, k)

    # Real scores
    ytr, ysr = _sanitize(labels, scores_real)
    auc_real = _safe_auc(ytr, ysr)
    mrr_real = _safe_mrr(ytr, ysr)
    ndcg5_real = _safe_ndcg(ytr, ysr, 5)
    ndcg10_real = _safe_ndcg(ytr, ysr, 10)

    # 70-news scores
    yt70, ys70 = _sanitize(labels, scores_70_news)
    auc_70 = _safe_auc(yt70, ys70)
    mrr_70 = _safe_mrr(yt70, ys70)
    ndcg5_70 = _safe_ndcg(yt70, ys70, 5)
    ndcg10_70 = _safe_ndcg(yt70, ys70, 10)

    return (
        auc_real,
        mrr_real,
        ndcg5_real,
        ndcg10_real,
        auc_70,
        mrr_70,
        ndcg5_70,
        ndcg10_70,
    )


# Diversity Metrics

def ILAD(vecs):
    # similarity = torch.mm(vecs, vecs.T)
    # similarity = cosine_similarity(X=vecs)
    similarity = F.cosine_similarity(vecs.unsqueeze(dim=0), vecs.unsqueeze(dim=1))
    distance = (1 - similarity)/2
    score = distance.mean()-1/distance.shape[0]
    return score.item()


def ILMD(vecs):
    # similarity = torch.mm(vecs, vecs.T)
    # similarity = cosine_similarity(X=vecs)
    similarity = F.cosine_similarity(vecs.unsqueeze(dim=0), vecs.unsqueeze(dim=1))
    distance = (1 - similarity) / 2
    score = distance.min()
    return score.item()

def density_ILxD(scores, news_emb, top_k=5):
    """
    Args:
        scores: [batch_size, y_pred_score]
        news_emb: [batch_size, news_num, news_emb_size]
        top_k: integer, n=5, n=10
    """
    top_ids = torch.argsort(scores)[-top_k:]
    news_emb =  news_emb / torch.sqrt(torch.square(news_emb).sum(dim=-1)).reshape((len(news_emb), 1))
    # nv: (top_k, news_emb_size)
    nv = news_emb[top_ids]
    ilad = ILAD(nv)
    ilmd = ILMD(nv)

    return ilad, ilmd



