"""
DeepGraphMut (DGM) — PyTorch Geometric implementation
Based on: "DeepGraphMut: a graph-based deep learning method for cancer
prognosis using somatic mutation profile"

Full pipeline
─────────────
  MutationDataset  →  DataLoader  →  DeepGraphMut
                                         │
                                    GNNEncoder  (2× TransformerConv + GraphNorm + ELU)
                                         │  node embeddings (N × d)
                                    global_mean_pool  →  patient embedding (B × d)
                                         │
                                    NodeDecoder  (MHA + Linear + BN + ELU + Linear)
                                         │  reconstruction logits (N,)
                                    FocalLoss  ←  original binary mutations
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphNorm, TransformerConv, global_mean_pool


# ══════════════════════════════════════════════════════════════════════════════
# Focal Loss
# ══════════════════════════════════════════════════════════════════════════════


class FocalLoss(nn.Module):
    """
    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

    Designed for somatic mutation data where typically < 5 % of genes are
    mutated per patient — a classic "many zeros" imbalance.

    gamma  > 0  down-weights easy (well-classified) negatives so the model
                focuses gradient on the rare positives.
    alpha        weights the positive class in the cross-entropy term.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (N,)  raw pre-sigmoid scores
        targets : (N,)  float labels  ∈ {0, 1}
        """
        probs = torch.sigmoid(logits)

        # standard per-sample BCE (numerically stable via logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t = probability assigned to the ground-truth class
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # α_t weights: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ══════════════════════════════════════════════════════════════════════════════
# GNN Encoder
# ══════════════════════════════════════════════════════════════════════════════


class GNNEncoder(nn.Module):
    """
    Two-layer Graph Transformer encoder over the PPI network.

    Layer 1 : TransformerConv(in → hidden × heads, concat=True)
              → GraphNorm  → ELU
    Layer 2 : TransformerConv(hidden × heads → out, heads=1, concat=False)
              (no normalisation / activation — raw embeddings passed downstream)

    TransformerConv implements multi-head attention-based message passing
    (Shi et al., 2021).  GraphNorm normalises within each graph in the
    batch, which is more appropriate than BatchNorm for variable-size graphs
    (Cai et al., 2021).
    """

    def __init__(
        self,
        in_channels: int,       # 1  (binary mutation flag per gene-node)
        hidden_channels: int,   # width per head before concatenation
        out_channels: int,      # final node embedding dimension  d
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Layer 1 — output width = hidden_channels × heads  (concat=True)
        self.conv1 = TransformerConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.norm1 = GraphNorm(hidden_channels * heads)
        self.act = nn.ELU()

        # Layer 2 — single head collapses to out_channels
        self.conv2 = TransformerConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

    def forward(
        self,
        x: torch.Tensor,            # (N_total, in_channels)
        edge_index: torch.Tensor,   # (2, E)
        batch: torch.Tensor,        # (N_total,)  graph index per node
    ) -> torch.Tensor:              # (N_total, out_channels)
        # ── layer 1 ──────────────────────────────────────────────────────────
        x = self.conv1(x, edge_index)   # (N, hidden × heads)
        x = self.norm1(x, batch)        # graph-wise feature normalisation
        x = self.act(x)

        # ── layer 2 ──────────────────────────────────────────────────────────
        x = self.conv2(x, edge_index)   # (N, out_channels)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Node-level Decoder
# ══════════════════════════════════════════════════════════════════════════════


class NodeDecoder(nn.Module):
    """
    Reconstructs per-node mutation probabilities from node embeddings.

    Pipeline
    ────────
    Multi-head self-attention  (global context across genes of each patient)
       ↓  residual  +  LayerNorm
    Linear(embed_dim → ff_dim)  →  BatchNorm1d  →  ELU
       ↓
    Linear(ff_dim → 1)          →  logit per gene node

    Because every patient graph contains exactly G genes (same topology),
    we reshape the flat batch tensor from (B·G, E) to (B, G, E) and run
    standard batched MHA — no Python loop needed, and attention stays
    within each patient (no cross-patient leakage).
    """

    def __init__(
        self,
        embed_dim: int,     # must equal GNNEncoder.out_channels
        num_heads: int = 2,
        ff_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention over the gene sequence of each patient.
        # batch_first=True  →  input/output shape: (B, G, E)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Feed-forward head
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.bn = nn.BatchNorm1d(ff_dim)
        self.act = nn.ELU()
        self.linear2 = nn.Linear(ff_dim, 1)     # one logit per gene

    def forward(
        self,
        x: torch.Tensor,                        # (N_total, embed_dim)
        batch: Optional[torch.Tensor] = None,   # (N_total,)  graph indices
    ) -> torch.Tensor:                          # (N_total,)  logits
        if batch is not None:
            B = int(batch.max().item()) + 1
            G = x.shape[0] // B                    # genes per patient (constant)
            x3 = x.view(B, G, -1)                  # (B, G, E)
        else:
            B, G = 1, x.shape[0]
            x3 = x.unsqueeze(0)                    # (1, G, E)

        # Multi-head self-attention — each patient attends over its own genes
        attn_out, _ = self.attn(x3, x3, x3)        # (B, G, E)
        x_attended = attn_out.reshape(B * G, -1)     # (N_total, E)

        # Residual connection + LayerNorm
        x = self.attn_norm(x + x_attended)          # (N_total, E)

        # Feed-forward block
        x = self.linear1(x)     # (N_total, ff_dim)
        x = self.bn(x)
        x = self.act(x)
        x = self.linear2(x)     # (N_total, 1)
        return x.squeeze(-1)    # (N_total,)


# ══════════════════════════════════════════════════════════════════════════════
# DeepGraphMut — full model
# ══════════════════════════════════════════════════════════════════════════════


class DeepGraphMut(nn.Module):
    """
    DeepGraphMut (DGM): graph autoencoder for cancer prognosis via somatic mutation.

    Input (per patient)
    ───────────────────
    edge_index : (2, E)   shared PPI network topology (protein–protein interactions)
    x          : (G, 1)   binary mutation flag per gene node

    Forward outputs
    ───────────────
    patient_emb : (B, embed_dim)  latent patient representation  →  prognosis tasks
    node_logits : (N_total,)      per-node reconstruction logits (sigmoid → mut. prob.)

    Parameters
    ──────────
    hidden_channels  width per attention head in layer 1 of the encoder
    embed_dim        node / patient embedding dimension  d  (≈ 10 in the paper)
    encoder_heads    number of attention heads in TransformerConv
    decoder_heads    number of attention heads in the decoder MHA
    ff_dim           hidden width of the decoder feed-forward block
    dropout          dropout probability (applied inside TransformerConv and MHA)
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        embed_dim: int = 10,
        encoder_heads: int = 4,
        decoder_heads: int = 2,
        ff_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = GNNEncoder(
            in_channels=1,
            hidden_channels=hidden_channels,
            out_channels=embed_dim,
            heads=encoder_heads,
            dropout=dropout,
        )
        self.decoder = NodeDecoder(
            embed_dim=embed_dim,
            num_heads=decoder_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, data: Data):
        """
        data : PyG Batch with attributes  .x  .edge_index  .batch

        Returns
        ───────
        patient_emb : (B, embed_dim)
        node_logits : (N_total,)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. GNN encoder → contextualised node embeddings
        node_emb = self.encoder(x, edge_index, batch)       # (N_total, d)

        # 2. Global mean pooling → one vector summarises each patient's graph
        patient_emb = global_mean_pool(node_emb, batch)     # (B, d)

        # 3. Decoder → reconstruct per-gene mutation probability
        node_logits = self.decoder(node_emb, batch)         # (N_total,)

        return patient_emb, node_logits

    # ── loss ──────────────────────────────────────────────────────────────────

    def compute_loss(self, data: Data) -> torch.Tensor:
        """Focal reconstruction loss against the original binary mutation flags."""
        _, node_logits = self.forward(data)
        targets = data.x.squeeze(-1).float()    # (N_total,)
        return self.loss_fn(node_logits, targets)

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_patients(self, data: Data) -> torch.Tensor:
        """Returns patient-level embeddings (no gradient tracking)."""
        self.eval()
        patient_emb, _ = self.forward(data)
        return patient_emb


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════


class MutationDataset(PyGDataset):
    """
    Converts a patient × gene binary mutation matrix into a PyG dataset.

    Each patient i becomes a Data object:
        x          : (G, 1)  float — binary mutation flags for each gene
        edge_index : (2, E)        — shared PPI topology (same for every patient)
        num_nodes  : G

    Parameters
    ──────────
    mutation_matrix : np.ndarray  shape (P, G)   P patients, G genes
    edge_index      : torch.LongTensor  shape (2, E)
    """

    def __init__(
        self,
        mutation_matrix: np.ndarray,
        edge_index: torch.Tensor,
        transform=None,
    ) -> None:
        super().__init__(root=None, transform=transform)

        matrix = torch.tensor(mutation_matrix, dtype=torch.float32)  # (P, G)
        G = matrix.shape[1]

        # Build one Data object per patient (shared edge_index reference)
        self._data_list: List[Data] = [
            Data(
                x=matrix[i].unsqueeze(-1),  # (G, 1)
                edge_index=edge_index,
                num_nodes=G,
            )
            for i in range(matrix.shape[0])
        ]

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Training utilities
# ══════════════════════════════════════════════════════════════════════════════


def _train_one_epoch(
    model: DeepGraphMut,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Single training epoch; returns mean loss over batches."""
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _eval_one_epoch(
    model: DeepGraphMut,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Validation epoch (no gradient updates); returns mean loss over batches."""
    model.eval()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        total += model.compute_loss(batch).item()
    return total / len(loader)


def train(
    model: DeepGraphMut,
    dataset: MutationDataset,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    log_every: int = 10,
    checkpoint_dir: Optional[Path] = None,
    resume: bool = True,
    val_split: float = 0.15,
    val_dataset: Optional[MutationDataset] = None,
) -> dict:
    """
    Trains the autoencoder end-to-end with Adam and Focal Loss.

    Validation
    ──────────
    val_dataset  explicit held-out dataset; takes priority over val_split
    val_split    fraction of `dataset` to hold out for validation (default 0.15);
                 ignored when val_dataset is provided or val_split <= 0

    Checkpointing
    ─────────────
    checkpoint_dir  path to save/load checkpoints; None = disabled
    resume          if True and a checkpoint exists, training continues from it

    Saves two files:
      checkpoint_latest.pt  — overwritten every log_every epochs
      checkpoint_best.pt    — overwritten whenever val loss improves

    Returns
    ───────
    history : dict with keys "train" and (optionally) "val" — per-epoch mean losses
    """
    from torch.utils.data import random_split

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = model.to(device)

    # ── build train / val split ───────────────────────────────────────────────
    if val_dataset is not None:
        train_dataset = dataset
    elif val_split > 0.0:
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"Train / val split: {n_train} / {n_val} patients")
    else:
        train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_history: List[float] = []
    val_history: List[float] = []
    start_epoch = 1
    best_val_loss = float("inf")

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_ckpt = checkpoint_dir / "checkpoint_latest.pt"
        best_ckpt = checkpoint_dir / "checkpoint_best.pt"

        if resume and latest_ckpt.exists():
            ckpt = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            train_history = ckpt.get("train_history", ckpt.get("history", []))
            val_history = ckpt.get("val_history", [])
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(
                f"Resumed from checkpoint: epoch {ckpt['epoch']}, "
                f"train loss {train_history[-1]:.6f}"
                + (f", val loss {val_history[-1]:.6f}" if val_history else "")
            )
    else:
        latest_ckpt = best_ckpt = None

    for epoch in range(start_epoch, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device)
        train_history.append(train_loss)

        if val_loader is not None:
            val_loss = _eval_one_epoch(model, val_loader, device)
            val_history.append(val_loss)
        else:
            val_loss = None

        if epoch == 1 or epoch % log_every == 0:
            msg = f"Epoch [{epoch:4d}/{epochs}]  train={train_loss:.6f}"
            if val_loss is not None:
                msg += f"  val={val_loss:.6f}"
            print(msg)

        monitor_loss = val_loss if val_loss is not None else train_loss

        if checkpoint_dir is not None:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_history": train_history,
                "val_history": val_history,
                "best_val_loss": best_val_loss,
            }
            if epoch % log_every == 0 or epoch == epochs:
                torch.save(state, latest_ckpt)
            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                state["best_val_loss"] = best_val_loss
                torch.save(state, best_ckpt)
                label = "val" if val_loss is not None else "train"
                print(f"  ✓ New best {label} loss {best_val_loss:.6f} — saved {best_ckpt.name}")

    return {"train": train_history, "val": val_history}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers for inference / embedding extraction
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def extract_embeddings(
    model: DeepGraphMut,
    dataset: MutationDataset,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Runs the full dataset through the encoder and returns all patient embeddings.

    Returns
    ───────
    embeddings : (P, embed_dim)  — one row per patient
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    parts: List[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        patient_emb, _ = model(batch)
        parts.append(patient_emb.cpu())

    return torch.cat(parts, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# Real-data loader
# ══════════════════════════════════════════════════════════════════════════════


def load_real_data(
    joined_parquet: str = "data/joined.parquet",
    ppi_txt: str = "data/HumanNet90_Symbol.txt",
) -> tuple[np.ndarray, torch.Tensor, list[str], list[str]]:
    """
    Loads real somatic mutation data and the HumanNet PPI network.

    Steps
    ─────
    1. Read mutation vectors from joined.parquet  (patients × genes)
    2. Read HumanNet90 PPI edges (gene symbol pairs)
    3. Restrict PPI edges to genes present in the mutation panel
    4. Build a PyG edge_index using per-gene integer indices

    Returns
    ───────
    mutation_matrix : np.ndarray  (P, G)   float32 binary
    edge_index      : torch.LongTensor  (2, E_filtered)
    gene_names      : list[str]  length G — ordered gene list
    primary_sites   : list[str]  length P — cancer type per patient
    """
    import pandas as pd

    df = pd.read_parquet(joined_parquet)

    # Gene order comes from the first row's Hugo_Symbol_mutation metadata
    gene_names: list[str] = list(df["Hugo_Symbol_mutation"].iloc[0])

    # Stack mutation vectors into (P, G) float32 matrix
    mutation_matrix = np.stack(df["mutation"].to_numpy()).astype(np.float32)
    # Binarise: any non-zero value → mutated
    mutation_matrix = (mutation_matrix > 0).astype(np.float32)

    primary_sites: list[str] = df["primary_site"].fillna("Unknown").tolist()

    # Build gene → index mapping
    gene2idx = {g: i for i, g in enumerate(gene_names)}

    # Load PPI and keep only edges where both endpoints are in the panel
    ppi = pd.read_csv(ppi_txt, sep="\t", header=None, names=["g1", "g2", "score"])
    mask = ppi["g1"].isin(gene2idx) & ppi["g2"].isin(gene2idx)
    ppi_filt = ppi[mask]

    src = ppi_filt["g1"].map(gene2idx).to_numpy()
    dst = ppi_filt["g2"].map(gene2idx).to_numpy()

    # Undirected: add both directions
    edge_index = torch.tensor(
        np.stack(
            [np.concatenate([src, dst]), np.concatenate([dst, src])],
            axis=0,
        ),
        dtype=torch.long,
    )

    print(
        f"Real data       |  patients={len(df)}  genes={len(gene_names)}\n"
        f"Mutation rate   |  {mutation_matrix.mean():.2%}\n"
        f"PPI edges       |  {len(ppi_filt)} pairs in panel  "
        f"({edge_index.shape[1]} directed edges)\n"
        f"Cancer types    |  {df['primary_site'].nunique()} unique sites"
    )
    return mutation_matrix, edge_index, gene_names, primary_sites


# ══════════════════════════════════════════════════════════════════════════════
# Example usage
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 1. Real TCGA/CCLE somatic mutation data ───────────────────────────────
    mutation_matrix, edge_index, gene_names, primary_sites = load_real_data()

    # ── 2. PyG dataset ────────────────────────────────────────────────────────
    dataset = MutationDataset(mutation_matrix, edge_index)
    print(f"\nDataset:  {len(dataset)} patients  |  {dataset[0]}")

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = DeepGraphMut(
        hidden_channels=32,
        embed_dim=10,
        encoder_heads=4,
        decoder_heads=2,
        ff_dim=32,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # ── 4. Training ───────────────────────────────────────────────────────────
    history = train(
        model=model,
        dataset=dataset,
        epochs=30,
        batch_size=64,
        lr=1e-4,
        log_every=5,
        val_split=0.15,
        checkpoint_dir=Path("checkpoints"),
    )

    print(f"\nFinal train loss: {history['train'][-1]:.6f}")
    if history["val"]:
        best_val = min(history["val"])
        best_ep = history["val"].index(best_val) + 1
        print(f"Final val   loss: {history['val'][-1]:.6f}")
        print(f"Best  val   loss: {best_val:.6f} (epoch {best_ep})")

    # ── 5. Patient embeddings ─────────────────────────────────────────────────
    embeddings = extract_embeddings(model, dataset, batch_size=128)
    print(f"\nPatient embeddings: {embeddings.shape}")
    print(f"Embedding stats:  mean={embeddings.mean():.4f}  std={embeddings.std():.4f}")
    print("Done — embeddings ready for survival analysis / clustering.")
