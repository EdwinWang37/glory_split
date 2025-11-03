"""
E2E smoke test for GLORYSplit validation path using the local MINDsmall data.

Notes
- Marked as slow and skipped by default. Enable by setting RUN_E2E=1.
- Forces dataset_lang to a non-English value to avoid loading GloVe file.
- Runs on CPU and only checks first 1-2 samples for shape and no exceptions.
"""
from pathlib import Path
import os
import sys

import numpy as np
import torch
import pytest
from omegaconf import OmegaConf


# Ensure src/ is importable
THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
SRC_ROOT = PROJ_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataload.data_load import load_data  # noqa: E402
from utils.common import load_model  # noqa: E402
from models.GLORYSplit import GLORYSplit  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E", "0") != "1",
    reason="Set RUN_E2E=1 to run slow E2E smoke test",
)


def _compose_min_cfg(num_workers: int):
    root = PROJ_ROOT.resolve()
    data_dir = root / "data" / "MINDsmall"
    cfg = OmegaConf.create({
        "path": {
            "root_dir": str(root),
            "data_dir": str(root / "data"),
            "ckp_dir": str(root / "checkpoint"),
            "glove_path": str(root / "data" / "glove" / "glove.840B.300d.txt"),
        },
        "dataset": {
            "dataset_name": "MINDsmall",
            "dataset_dir": str(data_dir),
            "train_dir": str(data_dir / "train"),
            "val_dir": str(data_dir / "val"),
            "test_dir": str(data_dir / "test"),
            # Avoid loading huge GloVe file
            "dataset_lang": "norwegian",
        },
        "model": {
            "model_name": "GLORYSplit",
            "use_graph": True,
            "use_entity": True,
            "directed": True,
            "num_neighbors": 8,
            "k_hops": 2,
            "entity_size": 5,
            "entity_neighbors": 10,
            "entity_emb_dim": 100,
            "word_emb_dim": 300,
            "head_num": 20,
            "head_dim": 20,
            "attention_hidden_dim": 200,
        },
        "npratio": 4,
        "batch_size": 32,
        "gpu_num": 1,
        "num_workers": num_workers,
        "dropout_probability": 0.2,
    })
    return cfg


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.mark.slow
def test_val_smoke_workers_0_and_2():
    device = _select_device()
    for nw in (0, 2):
        cfg = _compose_min_cfg(num_workers=nw)
        model = load_model(cfg).to(device)
        dl = load_data(cfg, mode="val", model=model, local_rank=0, device=device)

        module = getattr(model, "module", model)
        took = 0
        for i, (subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, labels) in enumerate(dl):
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(device, non_blocking=True)
            candidate_entity = candidate_entity.to(device, non_blocking=True)
            entity_mask = entity_mask.to(device, non_blocking=True)
            clicked_entity = clicked_entity.to(device, non_blocking=True)

            s_real, s_70 = module.validation_process(
                subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask
            )
            # Shapes: [1, num_candidates]
            assert s_real.dim() == 2 and s_70.dim() == 2
            assert s_real.shape[-1] == candidate_emb.shape[-2]
            assert s_70.shape[-1] == candidate_emb.shape[-2]

            took += 1
            if took >= 2:
                break

