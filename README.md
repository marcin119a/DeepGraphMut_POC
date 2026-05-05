# DeepGraphMut — POC

A proof-of-concept PyTorch Geometric implementation of **DeepGraphMut (DGM)**, a graph-based deep learning method for cancer prognosis from somatic mutation profiles.

> Based on: *"DeepGraphMut: a graph-based deep learning method for cancer prognosis using somatic mutation profile"*

## Overview

Each cancer patient is represented as a graph where **nodes are genes** and **edges are protein–protein interactions (PPI)** from the HumanNet90 network. Binary mutation flags (mutated / not mutated) serve as node features. A Graph Transformer autoencoder learns compact patient embeddings that are evaluated on a downstream survival prediction task.

The POC benchmarks DGM patient embeddings against **PyNBS** (network-propagated mutation profiles) using concordance index (C-index) from penalised Cox regression, per cancer type.

## Architecture

```
MutationDataset  →  DataLoader  →  DeepGraphMut
                                        │
                                   GNNEncoder
                                   ├─ TransformerConv (2 layers)
                                   ├─ GraphNorm + ELU
                                   └─ global_mean_pool → patient embedding (B × d)
                                        │
                                   NodeDecoder
                                   ├─ Multi-head self-attention (per patient)
                                   ├─ Residual + LayerNorm
                                   └─ Linear → reconstruction logits (N,)
                                        │
                                   FocalLoss ← binary mutation targets
```

**Key design choices:**
- **FocalLoss** (α=0.25, γ=2) handles the severe class imbalance (~5% of genes mutated per patient).
- **GraphNorm** normalises within each patient graph — more appropriate than BatchNorm for variable-size graphs.
- **TransformerConv** implements multi-head attention-based message passing (Shi et al., 2021).

## Repository structure

```
.
├── deep_graph_mut.py          # Core model (DGM, GNNEncoder, NodeDecoder, FocalLoss, Dataset)
├── survival_downstream.py     # C-index benchmark: DGM vs PyNBS
├── requirements.txt
├── scripts/
│   ├── download_data.sh       # Download TCGA + CCLE data (Kaggle + Google Drive)
│   ├── download_survival.py   # Download TCGA Pan-Cancer survival data (UCSC Xena)
│   ├── preprocessing.py       # Export expression/mutation matrices for downstream tools
│   └── propagate_profile.py   # Standalone network propagation (PyNBS baseline)
└── data/
    ├── HumanNet90_Symbol.txt       # PPI edge list (gene symbol pairs + score)
    ├── joined.parquet              # Merged mutation + expression + primary_site per patient
    ├── survival.parquet            # TCGA OS/DSS/PFI survival labels
    ├── dgm_embeddings.parquet      # Cached DGM patient embeddings
    ├── pynbs_embeddings.parquet    # Cached PyNBS propagated profiles
    ├── cindex_comparison.csv       # Per-cancer-type C-index results
    └── cindex_comparison.png       # Bar chart (DGM vs PyNBS)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `pyNBS` is installed directly from GitHub (`marcin119a/pyNBS`). PyTorch Geometric requires compatible `torch` and CUDA versions — see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if needed.

## Quickstart

### 1. Download data

```bash
bash scripts/download_data.sh          # TCGA mutations + CCLE expression → data/joined.parquet
python scripts/download_survival.py    # TCGA Pan-Cancer OS/DSS/PFI → data/survival.parquet
```

### 2. Train DGM and compare against PyNBS

```bash
python survival_downstream.py \
    --epochs 50 \
    --embed-dim 10 \
    --checkpoint-dir checkpoints
```

Results are written to `data/cindex_comparison.csv` and `data/cindex_comparison.png`.

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 50 | DGM training epochs |
| `--embed-dim` | 10 | Patient embedding dimension |
| `--alpha` | 0.7 | PyNBS network propagation coefficient |
| `--pynbs-genes` | `panel` | Gene scope for PyNBS: `panel` (same as DGM), `1hop`, or `full` |
| `--min-patients` | 20 | Min patients with OS data per cancer type |
| `--checkpoint-dir` | — | Save/resume DGM training checkpoints |
| `--skip-dgm` | — | Load cached DGM embeddings instead of retraining |
| `--skip-pynbs` | — | Load cached PyNBS embeddings |

### 3. Use the model directly

```python
from deep_graph_mut import DeepGraphMut, MutationDataset, load_real_data, train, extract_embeddings

mutation_matrix, edge_index, gene_names, primary_sites = load_real_data()
dataset = MutationDataset(mutation_matrix, edge_index)

model = DeepGraphMut(hidden_channels=32, embed_dim=10, encoder_heads=4)
history = train(model, dataset, epochs=50, batch_size=64, lr=1e-4)

embeddings = extract_embeddings(model, dataset)  # (P, embed_dim)
```

## Data sources

| Dataset | Source |
|---|---|
| TCGA somatic mutations + RNA-seq | [Kaggle — martininf1n1ty/rna-mutations-all-datasets](https://www.kaggle.com/datasets/martininf1n1ty/rna-mutations-all-datasets) |
| CCLE prepared profiles | Google Drive (downloaded by `download_data.sh`) |
| HumanNet90 PPI network | Bundled in `data/HumanNet90_Symbol.txt` |
| TCGA Pan-Cancer survival | UCSC Xena Pan-Can Atlas Hub (Liu et al. 2018) |

## Network propagation utility

`scripts/propagate_profile.py` is a standalone tool for propagating any mutation/expression profile over a PPI network:

```bash
python scripts/propagate_profile.py \
    data/COADREAD/features_mut_list.txt \
    data/HumanNet90_Symbol.txt \
    --fmt list -o propagated.tsv -v
```

Supported input formats: `matrix`, `matrix-T`, `list` (sample–gene pairs).

## Dependencies

- `torch` + `torch_geometric` — GNN model
- `scikit-survival`, `lifelines` — survival analysis / C-index
- `scikit-learn` — PCA, StandardScaler
- `pandas`, `numpy` — data handling
- `kagglehub`, `gdown` — data download
- `pyNBS` — network propagation baseline
- `matplotlib` — result plots
