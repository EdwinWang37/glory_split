"""
Unit tests for GLORYSplit shape/typing robustness around validation_process and
client.process_news. These tests do not require any dataset files and run fully
on CPU in milliseconds.

Covers:
- mapping_idx being 1D vs [1, N] 2D
- candidate_news as pre-encoded [N, D] vs [1, N, D]
- mapping_idx is not modified in-place (clone safety)
- validation path when subgraph.x is already encoded floats
"""
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data


# Ensure src/ is importable when running via pytest from repo root
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.GLORYSplit import GLORYSplit  # noqa: E402


def _make_cfg(use_entity: bool = True):
    return OmegaConf.create({
        "model": {
            "model_name": "GLORYSplit",
            "use_graph": True,
            "use_entity": use_entity,
            "head_num": 20,
            "head_dim": 20,
            "word_emb_dim": 300,
            "attention_hidden_dim": 200,
            "entity_size": 5,
            "entity_neighbors": 10,
            "entity_emb_dim": 100,
        },
        "dataset": {"dataset_lang": "english"},
        "dropout_probability": 0.0,
    })


def test_validation_process_shapes_and_clone_safety():
    cfg = _make_cfg(use_entity=True)

    news_dim = cfg.model.head_num * cfg.model.head_dim  # 400
    # Minimal embeddings: small vocab and entity table
    glove = np.random.randn(10, cfg.model.word_emb_dim).astype(np.float32)
    entity_emb = np.random.randn(50, cfg.model.entity_emb_dim).astype(np.float32)

    model = GLORYSplit(cfg, glove_emb=glove, entity_emb=entity_emb)

    # Subgraph with already-encoded node features (validation path)
    N = 64
    x = torch.randn(N, news_dim)
    edge_index = torch.randint(0, N, (2, 256))
    subgraph = Data(x=x, edge_index=edge_index)

    # 1D mapping (with -1 as padding that should be masked)
    m1 = torch.tensor([-1, 2, 5, 9, 11, -1, 3, 7, 8, -1], dtype=torch.long)
    m1_before = m1.clone()

    # Candidate news as pre-encoded vectors: [N, D] and [1, N, D]
    C = 6
    cand2d = torch.randn(C, news_dim)
    cand3d = cand2d.unsqueeze(0)

    # Entity inputs (indices)
    # clicked entities: [num_clicked, entity_size]
    clicked_ent = torch.randint(0, 50, (len(m1), cfg.model.entity_size), dtype=torch.long)

    # candidate_entity: concat(origin_entity, neighbor_entity)
    origin = torch.randint(0, 50, (C, cfg.model.entity_size), dtype=torch.long)
    neighbors = torch.randint(
        0, 50, (C, cfg.model.entity_size * cfg.model.entity_neighbors), dtype=torch.long
    )
    cand_entity = torch.cat([origin, neighbors], dim=-1)
    # Simple mask: 1 where neighbor id > 0
    entity_mask = (neighbors > 0).to(torch.long)

    # 2D candidate
    s_real_2d, s_70_2d = model.validation_process(
        subgraph, m1, clicked_ent, cand2d, cand_entity, entity_mask
    )
    # 3D candidate
    s_real_3d, s_70_3d = model.validation_process(
        subgraph, m1, clicked_ent, cand3d, cand_entity, entity_mask
    )

    # Basic shape checks: [B=1, C]
    assert s_real_2d.shape[-1] == C and s_70_2d.shape[-1] == C
    assert s_real_3d.shape[-1] == C and s_70_3d.shape[-1] == C

    # 2D/3D candidate inputs should be equivalent
    assert torch.allclose(s_real_2d, s_real_3d, atol=1e-5)
    assert torch.allclose(s_70_2d, s_70_3d, atol=1e-5)

    # Mapping clone safety: original tensor unchanged
    assert torch.equal(m1, m1_before)

    # 2D mapping ([1, N]) should be equivalent to 1D
    m2 = m1.unsqueeze(0)
    s_real_2d_b, s_70_2d_b = model.validation_process(
        subgraph, m2, clicked_ent, cand2d, cand_entity, entity_mask
    )
    assert torch.allclose(s_real_2d, s_real_2d_b, atol=1e-5)
    assert torch.allclose(s_70_2d, s_70_2d_b, atol=1e-5)

