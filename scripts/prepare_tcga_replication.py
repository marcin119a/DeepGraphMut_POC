#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert data_replication/ → data_tcga/ in the format expected by
survival_downstream.py and deep_graph_mut.load_real_data().

Outputs
-------
  data_tcga/joined.parquet    – one row per patient:
                                 case_barcode, mutation (list), Hugo_Symbol_mutation (list), primary_site
  data_tcga/NCG_network.txt   – headerless TSV: gene_a <TAB> gene_b <TAB> 1.0

Usage
-----
  python scripts/prepare_tcga_replication.py [--replication-dir data_replication] [--out-dir data_tcga]
  python survival_downstream.py \
    --data-dir data_tcga \
    --out-dir data_tcga \
    --ppi-file NCG_network.txt \
    --checkpoint-dir checkpoints_tcga \
    --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CANCER_PROJECTS = [
    "TCGA-BLCA", "TCGA-LGG",  "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD",
    "TCGA-GBM",  "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LIHC",
    "TCGA-LUAD", "TCGA-LUSC", "TCGA-OV",   "TCGA-SKCM", "TCGA-STAD",
    "TCGA-UCEC",
]

META_COLS = {"primary_site", "project_short_name"}


def prepare_joined(mut_path: Path, out_path: Path) -> list[str]:
    print(f"Reading {mut_path} …")
    df = pd.read_parquet(mut_path)
    print(f"  Full shape: {df.shape}")

    df = df[df["project_short_name"].isin(CANCER_PROJECTS)].copy()
    print(f"  After TCGA filter: {df.shape[0]} patients")
    print(df["project_short_name"].value_counts().to_string())

    gene_cols = [c for c in df.columns if c not in META_COLS]
    gene_names = gene_cols

    # OR-merge duplicate patient IDs (keep max across possible duplicates)
    df.index.name = "case_barcode"
    df = df.reset_index()
    dup_mask = df.duplicated(subset=["case_barcode"], keep=False)
    if dup_mask.any():
        print(f"  Merging {dup_mask.sum()} duplicate barcodes …")
        meta = df[["case_barcode", "primary_site", "project_short_name"]].drop_duplicates("case_barcode")
        gene_df = df[["case_barcode"] + gene_cols].groupby("case_barcode").max().reset_index()
        df = meta.merge(gene_df, on="case_barcode")

    mut_matrix = df[gene_cols].values.astype(np.float32)
    mut_matrix = (mut_matrix > 0).astype(np.float32)

    records = []
    for i, row in enumerate(df.itertuples(index=False)):
        records.append({
            "case_barcode": row.case_barcode,
            "mutation": mut_matrix[i].tolist(),
            "Hugo_Symbol_mutation": gene_names,
            "primary_site": row.project_short_name,
        })

    out_df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path}  ({len(out_df)} patients, {len(gene_names)} genes)")
    return gene_names


def prepare_network(edges_path: Path, out_path: Path) -> None:
    print(f"\nReading {edges_path} …")
    edges = pd.read_csv(edges_path, sep="\t")
    print(f"  Edges: {len(edges):,}")

    # Add dummy score column expected by load_real_data and pyNBS
    edges["score"] = 1.0
    edges = edges.rename(columns={"gene_a": "g1", "gene_b": "g2"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges[["g1", "g2", "score"]].to_csv(out_path, sep="\t", index=False, header=False)
    print(f"Saved {out_path}  ({len(edges):,} edges)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replication-dir", default="data_replication")
    parser.add_argument("--out-dir",         default="data_tcga")
    args = parser.parse_args()

    rep_dir = Path(args.replication_dir)
    out_dir = Path(args.out_dir)

    prepare_joined(
        rep_dir / "mutation_binary_matrix.parquet",
        out_dir / "joined.parquet",
    )
    prepare_network(
        rep_dir / "NCG_network_edges.tsv",
        out_dir / "NCG_network.txt",
    )
    print("\nDone. Next steps:")
    print("  python scripts/download_survival.py --data-dir data_tcga")
    print("  python survival_downstream.py --data-dir data_tcga --out-dir data_tcga \\")
    print("    --checkpoint-dir checkpoints_tcga --epochs 50")


if __name__ == "__main__":
    main()
