#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised survival downstream task: C-index comparison
DGM (DeepGraphMut) vs PyNBS (network-propagated mutation profiles).

Pipeline
--------
1. Build DGM embeddings from joined.parquet + HumanNet90 PPI
   (trains the autoencoder, then extracts patient-level embeddings)
2. Build PyNBS embeddings via network propagation
   (calls propagate_profile logic directly — no subprocess needed)
3. Merge both embedding sets with survival.parquet
4. Per cancer type: fit penalised Cox PH on each embedding set, compute C-index
5. Plot Figure-7-style grouped bar chart

Usage
-----
  python survival_downstream.py [options]

  --epochs        DGM training epochs (default 50)
  --embed-dim     DGM embedding dimension (default 10)
  --alpha         PyNBS propagation coefficient (default 0.7)
  --min-patients  min patients with OS data required per cancer type (default 20)
  --out-dir       output directory for plot + CSV (default data)
  --checkpoint-dir  directory to save/resume DGM training checkpoints (default: no checkpointing)
  --no-resume     disable resuming from an existing checkpoint
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# C-index via penalised Cox (scikit-survival)
# ─────────────────────────────────────────────────────────────────────────────

def _cindex(X: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """
    Fit a penalised Cox PH on X and return concordance index.
    Falls back to Harrell's concordance on the first principal component
    when the Cox solver fails (too few events, collinear features, etc.).
    """
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    y = np.array([(bool(e), t) for e, t in zip(event, time)],
                 dtype=[("event", bool), ("time", float)])

    X_sc = StandardScaler().fit_transform(X)

    # Reduce to at most 10 PCs when more features than samples
    n_comp = min(10, X_sc.shape[1], X_sc.shape[0] - 1)
    if n_comp < X_sc.shape[1]:
        X_sc = PCA(n_components=n_comp).fit_transform(X_sc)

    try:
        cox = CoxnetSurvivalAnalysis(l1_ratio=0.5, max_iter=10_000, fit_baseline_model=True)
        cox.fit(X_sc, y)
        risk = cox.predict(X_sc)
    except Exception:
        # fallback: use PC-1 risk score
        risk = X_sc[:, 0]

    ci, *_ = concordance_index_censored(y["event"], y["time"], risk)
    return float(ci)


# ─────────────────────────────────────────────────────────────────────────────
# DGM embeddings
# ─────────────────────────────────────────────────────────────────────────────

def build_dgm_embeddings(
    joined_parquet: Path,
    ppi_txt: Path,
    epochs: int,
    embed_dim: int,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Returns DataFrame (index=case_barcode, cols=dgm_0..dgm_{d-1})."""
    import torch
    from deep_graph_mut import (
        DeepGraphMut, MutationDataset, load_real_data, train, extract_embeddings
    )

    print("─── DGM: loading data ───")
    mut_mat, edge_index, gene_names, primary_sites = load_real_data(
        str(joined_parquet), str(ppi_txt)
    )

    dataset = MutationDataset(mut_mat, edge_index)
    model = DeepGraphMut(
        hidden_channels=32,
        embed_dim=embed_dim,
        encoder_heads=4,
        decoder_heads=2,
        ff_dim=32,
        dropout=0.1,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"─── DGM: training {epochs} epochs ───")
    train(
        model, dataset,
        epochs=epochs,
        batch_size=64,
        lr=1e-4,
        log_every=10,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )

    print("─── DGM: extracting embeddings ───")
    emb = extract_embeddings(model, dataset, batch_size=128).numpy()  # (P, d)

    df = pd.read_parquet(joined_parquet, columns=["case_barcode"])
    barcodes = df["case_barcode"].apply(_patient_id).tolist()

    cols = [f"dgm_{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, index=barcodes, columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# PyNBS embeddings  (network-propagated mutation profiles)
# ─────────────────────────────────────────────────────────────────────────────

def build_pynbs_embeddings(
    joined_parquet: Path,
    ppi_txt: Path,
    alpha: float,
    gene_scope: str = "panel",
) -> pd.DataFrame:
    """Returns DataFrame (index=case_barcode, cols=network genes after propagation).

    gene_scope : "panel"  — restrict network to panel genes only (fair vs DGM)
                 "1hop"   — panel genes + their direct neighbours in HumanNet
                 "full"   — entire HumanNet (original behaviour, slow)
    """
    from pyNBS import data_import_tools as dit
    from pyNBS import network_propagation as prop
    from pyNBS import pyNBS_core as core

    print("─── PyNBS: loading mutation matrix ───")
    df = pd.read_parquet(joined_parquet)
    barcodes = df["case_barcode"].apply(_patient_id).tolist()
    gene_names: list[str] = list(df["Hugo_Symbol_mutation"].iloc[0])
    mut_mat = np.stack(df["mutation"].to_numpy()).astype(np.float32)
    mut_mat = (mut_mat > 0).astype(float)

    sm_mat = pd.DataFrame(mut_mat, index=barcodes, columns=gene_names)
    # OR-merge duplicate patient IDs (multiple samples per patient share the same 3-part barcode)
    sm_mat = sm_mat.groupby(level=0).max()

    print("─── PyNBS: loading network ───")
    network = dit.load_network_file(str(ppi_txt), delimiter="\t", verbose=False)
    network_nodes = list(network.nodes)
    print(f"  Network (full): {len(network_nodes)} nodes, {network.number_of_edges()} edges")

    overlap = set(sm_mat.columns) & set(network_nodes)
    print(f"  Gene overlap: {len(overlap)} / {len(sm_mat.columns)}")

    # ── Restrict network to requested gene scope ───────────────────────────────
    if gene_scope == "panel":
        keep = [n for n in network_nodes if n in set(sm_mat.columns)]
        network = network.subgraph(keep).copy()
    elif gene_scope == "1hop":
        panel_in_net = set(sm_mat.columns) & set(network_nodes)
        neighbours: set[str] = set()
        for g in panel_in_net:
            neighbours.update(network.neighbors(g))
        keep = list(panel_in_net | neighbours)
        network = network.subgraph(keep).copy()
    # "full" → no filtering

    # Drop isolated nodes — degree-0 nodes cause division by zero in the
    # adjacency normalisation, producing NaN kernels and collapsed embeddings.
    isolated = [n for n in network.nodes if network.degree(n) == 0]
    if isolated:
        network = network.subgraph([n for n in network.nodes if network.degree(n) > 0]).copy()

    network_nodes = list(network.nodes)
    print(f"  Network ({gene_scope}): {len(network_nodes)} nodes, {network.number_of_edges()} edges"
          + (f"  (dropped {len(isolated)} isolated)" if isolated else ""))

    # Align sm_mat columns to (possibly filtered) network; absent genes get 0
    sm_mat = sm_mat.reindex(columns=network_nodes, fill_value=0)

    print(f"─── PyNBS: computing kernel (alpha={alpha}) ───")
    network_I = pd.DataFrame(
        np.identity(len(network_nodes)),
        index=network_nodes, columns=network_nodes
    )
    kernel = prop.network_propagation(
        network, network_I, alpha=alpha, symmetric_norm=False, verbose=False
    )

    print("─── PyNBS: propagating ───")
    prop_data = prop.network_kernel_propagation(
        network, kernel, sm_mat, verbose=False
    )  # rows=samples, cols=network genes
    prop_data = core.qnorm(prop_data)

    prop_data.columns = [f"pynbs_{g}" for g in prop_data.columns]
    print(f"  Propagated shape: {prop_data.shape}")
    return prop_data


# ─────────────────────────────────────────────────────────────────────────────
# Survival C-index per cancer type
# ─────────────────────────────────────────────────────────────────────────────

def compute_cindex_per_cancer(
    embeddings: pd.DataFrame,
    survival: pd.DataFrame,
    min_patients: int,
) -> dict[str, float]:
    merged = embeddings.join(survival.set_index("case_barcode"), how="inner")
    merged = merged.dropna(subset=["OS", "OS.time"])
    merged = merged[merged["OS.time"] > 0]

    results: dict[str, float] = {}
    for ct, grp in merged.groupby("cancer_type"):
        if len(grp) < min_patients or grp["OS"].sum() < 5:
            continue
        X = grp[embeddings.columns].values
        t = grp["OS.time"].values
        e = grp["OS"].values.astype(bool)
        try:
            ci = _cindex(X, t, e)
            results[str(ct)] = ci
        except Exception as exc:
            print(f"  [{ct}] skipped: {exc}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    dgm_ci: dict[str, float],
    pynbs_ci: dict[str, float],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cancer_types = sorted(set(dgm_ci) | set(pynbs_ci))
    dgm_vals   = [dgm_ci.get(ct, np.nan)   for ct in cancer_types]
    pynbs_vals = [pynbs_ci.get(ct, np.nan) for ct in cancer_types]

    x = np.arange(len(cancer_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(cancer_types) * 0.7), 5))
    bars1 = ax.bar(x - width / 2, dgm_vals,   width, label="DGM",   color="#2563EB", alpha=0.85)
    bars2 = ax.bar(x + width / 2, pynbs_vals, width, label="PyNBS", color="#F59E0B", alpha=0.85)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xlabel("Cancer type")
    ax.set_ylabel("C-index (Cox PH)")
    ax.set_title("Supervised survival task: DGM vs PyNBS")
    ax.set_xticks(x)
    ax.set_xticklabels(cancer_types, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _patient_id(barcode: str) -> str:
    parts = str(barcode).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(barcode)[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",        default="data")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--embed-dim",       type=int,   default=10)
    parser.add_argument("--alpha",           type=float, default=0.7)
    parser.add_argument("--pynbs-genes",     default="panel",
                        choices=["panel", "1hop", "full"],
                        help="Gene scope for PyNBS network: panel=same as DGM (default), "
                             "1hop=panel+neighbours, full=entire HumanNet (slow)")
    parser.add_argument("--min-patients",    type=int,   default=20)
    parser.add_argument("--out-dir",         default="data")
    parser.add_argument("--skip-dgm",        action="store_true",
                        help="Skip DGM (use cached embeddings if present)")
    parser.add_argument("--skip-pynbs",      action="store_true",
                        help="Skip PyNBS (use cached embeddings if present)")
    parser.add_argument("--checkpoint-dir",  default=None,
                        help="Directory for DGM training checkpoints. "
                             "Training resumes automatically if a checkpoint exists.")
    parser.add_argument("--no-resume",       action="store_true",
                        help="Start DGM training from scratch even if a checkpoint exists.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joined_parquet   = data_dir / "joined.parquet"
    ppi_txt          = data_dir / "HumanNet90_Symbol.txt"
    survival_parquet = data_dir / "survival.parquet"

    for p in [joined_parquet, ppi_txt, survival_parquet]:
        if not p.exists():
            raise FileNotFoundError(f"Brak: {p}")

    survival = pd.read_parquet(survival_parquet)

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    # ── DGM embeddings ────────────────────────────────────────────────────────
    dgm_cache = out_dir / "dgm_embeddings.parquet"
    if args.skip_dgm and dgm_cache.exists():
        print(f"Loading cached DGM embeddings: {dgm_cache}")
        dgm_emb = pd.read_parquet(dgm_cache)
    else:
        dgm_emb = build_dgm_embeddings(
            joined_parquet, ppi_txt,
            epochs=args.epochs,
            embed_dim=args.embed_dim,
            checkpoint_dir=checkpoint_dir,
            resume=not args.no_resume,
        )
        dgm_emb.to_parquet(dgm_cache)
        print(f"Cached DGM embeddings: {dgm_cache}")

    # ── PyNBS embeddings ──────────────────────────────────────────────────────
    pynbs_cache = out_dir / "pynbs_embeddings.parquet"
    if args.skip_pynbs and pynbs_cache.exists():
        print(f"Loading cached PyNBS embeddings: {pynbs_cache}")
        pynbs_emb = pd.read_parquet(pynbs_cache)
    else:
        pynbs_emb = build_pynbs_embeddings(joined_parquet, ppi_txt, args.alpha, args.pynbs_genes)
        pynbs_emb.to_parquet(pynbs_cache)
        print(f"Cached PyNBS embeddings: {pynbs_cache}")

    # ── C-index per cancer type ───────────────────────────────────────────────
    print("\n─── Computing C-index: DGM ───")
    dgm_ci = compute_cindex_per_cancer(dgm_emb, survival, args.min_patients)

    print("\n─── Computing C-index: PyNBS ───")
    pynbs_ci = compute_cindex_per_cancer(pynbs_emb, survival, args.min_patients)

    # ── Results table ─────────────────────────────────────────────────────────
    all_ct = sorted(set(dgm_ci) | set(pynbs_ci))
    results_df = pd.DataFrame({
        "cancer_type": all_ct,
        "DGM":   [dgm_ci.get(ct, np.nan)   for ct in all_ct],
        "PyNBS": [pynbs_ci.get(ct, np.nan) for ct in all_ct],
    })
    results_df["DGM_wins"] = results_df["DGM"] > results_df["PyNBS"]

    csv_path = out_dir / "cindex_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))

    n_dgm_wins = results_df["DGM_wins"].sum()
    print(f"\nDGM wins: {n_dgm_wins}/{len(all_ct)} cancer types")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(dgm_ci, pynbs_ci, out_dir / "cindex_comparison.png")


if __name__ == "__main__":
    main()
