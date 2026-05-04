#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd


def _flatten_to_list(value):
    if isinstance(value, np.ndarray):
        return _flatten_to_list(value.tolist())
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten_to_list(item))
        return out
    return [value]


def _as_list_column(series: pd.Series) -> list:
    return [list(_flatten_to_list(v)) for v in series]


def _infer_sample_type(barcode: str) -> str:
    return "tumor" if str(barcode).startswith("TCGA") else "cell"


def _clean_subtype(value) -> str:
    if value is None:
        return "None"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "None"
    return text


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    out_dir = data_dir / "COADREAD"
    out_dir.mkdir(parents=True, exist_ok=True)

    joined_path = data_dir / "joined.parquet"
    ccle_path = data_dir / "ccle_filtered_merged_df.parquet"

    if not joined_path.exists():
        raise FileNotFoundError(
            f"Brak pliku {joined_path}. Najpierw uruchom scripts/download_data.sh."
        )
    if not ccle_path.exists():
        raise FileNotFoundError(
            f"Brak pliku {ccle_path}. Najpierw uruchom scripts/download_data.sh."
        )

    joined = pd.read_parquet(joined_path)
    ccle = pd.read_parquet(ccle_path)

    if not {"case_barcode", "expression", "mutation"}.issubset(joined.columns):
        raise ValueError(
            "joined.parquet musi miec kolumny: case_barcode, expression, mutation."
        )

    # Gene names: prefer lists embedded in joined.parquet, fallback to CCLE metadata
    expr_genes = []
    if "Hugo_Symbol_expression" in joined.columns:
        expr_genes = [
            str(x) for x in _flatten_to_list(joined.iloc[0]["Hugo_Symbol_expression"])
        ]
        expr_genes = [x for x in expr_genes if x and x.lower() != "nan"]
        expr_genes = list(dict.fromkeys(expr_genes))
    if not expr_genes:
        expr_genes = [
            str(x) for x in _flatten_to_list(ccle.iloc[0]["Hugo_Symbol_expression"])
        ]
        expr_genes = [x for x in expr_genes if x and x.lower() != "nan"]
        expr_genes = list(dict.fromkeys(expr_genes))

    # Mutation gene names:
    # prefer symbols embedded in joined.parquet; fallback to CCLE metadata
    mut_genes = []
    if "Hugo_Symbol_mutation" in joined.columns:
        mut_genes = [
            str(x) for x in _flatten_to_list(joined.iloc[0]["Hugo_Symbol_mutation"])
        ]
        mut_genes = [x for x in mut_genes if x and x.lower() != "nan"]
        mut_genes = list(dict.fromkeys(mut_genes))
    if not mut_genes and "Hugo_Symbol_mutation" in ccle.columns:
        mut_genes = [
            str(x) for x in _flatten_to_list(ccle.iloc[0]["Hugo_Symbol_mutation"])
        ]
        mut_genes = [x for x in mut_genes if x and x.lower() != "nan"]
        mut_genes = list(dict.fromkeys(mut_genes))

    sample_ids = joined["case_barcode"].astype(str).tolist()
    expr_vectors = _as_list_column(joined["expression"])
    mut_vectors = _as_list_column(joined["mutation"])

    if not expr_vectors or not mut_vectors:
        raise ValueError("joined.parquet jest pusty - brak danych do eksportu.")

    expr_len = len(expr_vectors[0])
    mut_len = len(mut_vectors[0])

    if len(expr_genes) != expr_len:
        expr_genes = [f"expr_gene_{i+1}" for i in range(expr_len)]
    if len(mut_genes) != mut_len:
        mut_genes = [f"mut_gene_{i+1}" for i in range(mut_len)]

    expr_matrix = pd.DataFrame(expr_vectors, index=sample_ids, columns=expr_genes).T
    mut_matrix = pd.DataFrame(mut_vectors, index=sample_ids, columns=mut_genes).T

    # Keep the same table layout as existing fake-data generator (index written to first column)
    expr_matrix.index.name = "barcode"
    mut_matrix.index.name = "barcode"

    labels = pd.DataFrame(
        {
            "barcode": sample_ids,
            "subtype": (
                joined["primary_site"].map(_clean_subtype).tolist()
                if "primary_site" in joined.columns
                else ["None"] * len(sample_ids)
            ),
            "type": [_infer_sample_type(x) for x in sample_ids],
        }
    )
    labels.index = labels["barcode"]

    expr_out = out_dir / "features_exp.txt"
    mut_out = out_dir / "features_mut_cnv_comb.txt"
    mut_list_out = out_dir / "features_mut_list.txt"
    labels_out = out_dir / "dataset_labels.txt"

    expr_matrix.to_csv(expr_out, sep="\t")
    mut_matrix.to_csv(mut_out, sep="\t")

    # Extra mutation file in 'list' format for propagate_profile.py:
    # sample<TAB>gene for every altered gene (value > 0).
    pairs = []
    for sample_id, values in zip(sample_ids, mut_vectors):
        for gene, value in zip(mut_genes, values):
            try:
                is_mutated = float(value) > 0.0
            except (TypeError, ValueError):
                is_mutated = False
            if is_mutated:
                pairs.append((sample_id, gene))
    pd.DataFrame(pairs, columns=["sample", "gene"]).to_csv(
        mut_list_out, sep="\t", index=False, header=False
    )

    labels.to_csv(labels_out, sep="\t", index=False)

    print(f"Zapisano: {expr_out}")
    print(f"Zapisano: {mut_out}")
    print(f"Zapisano: {mut_list_out}")
    print(f"Zapisano: {labels_out}")
    print(f"Liczba probek: {len(sample_ids)}")
    print(f"Wymiar expression: {expr_matrix.shape[0]} x {expr_matrix.shape[1]}")
    print(f"Wymiar DNA: {mut_matrix.shape[0]} x {mut_matrix.shape[1]}")


if __name__ == "__main__":
    main()
