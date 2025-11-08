# ✨GLORY: Global Graph-Enhanced Personalized News Recommendations
Code for our paper [_Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations_](https://arxiv.org/pdf/2307.06576.pdf) published at RecSys 2023. 

<p align="center">
  <img src="glory.jpg" alt="Glory Model Illustration" width="600" />
  <br>
  Glory Model Illustration
</p>


### Environment
> Python 3.8.10
> pytorch 1.13.1+cu117
```shell
cd GLORY

apt install unzip python3.8-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```shell
# dataset download
bash scripts/data_download.sh

# Run
python3 src/main.py model=GLORY dataset=MINDsmall reprocess=True
```

### Run on macOS (CPU/MPS)

This repo now supports single‑process training on macOS without NVIDIA GPUs.

- Install tools: `brew install wget unzip`.
- Create env (recommend Python 3.9+):
  - `python3 -m venv .venv && source .venv/bin/activate`
  - Install PyTorch for CPU/MPS following the official instructions for your macOS/Apple Silicon.
  - Install PyG (torch-geometric) matching your PyTorch version (see the official PyG install guide).
  - `pip install -r requirements.txt` (after torch/pyg are installed).
- Download data: `bash scripts/data_download.sh`.
- Run single‑process training (CPU/MPS auto‑detected):
  - `python3 src/main.py model=GLORY dataset=MINDsmall device=cpu reprocess=True`
  - For Apple Silicon try MPS: `python3 src/main.py model=GLORY dataset=MINDsmall device=mps reprocess=True`

Notes:
- No NCCL/DDP on macOS; training runs single‑process and can be slower.
- If DataLoader workers cause issues on macOS, override with `num_workers=0` via CLI.
- Set `export PROJECT_ROOT=$(pwd)` if config cannot resolve project root (usually auto‑detected).

### Bibliography

```shell
@misc{yang2023going,
      title={Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations}, 
      author={Boming Yang and Dairui Liu and Toyotaro Suzumura and Ruihai Dong and Irene Li},
      year={2023},
      publisher ={RecSys},
}
```

