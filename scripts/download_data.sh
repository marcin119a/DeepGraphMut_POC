#!/usr/bin/env bash
set -euo pipefail


python - << 'PY'
# -*- coding: utf-8 -*-
import ast
import os
from pathlib import Path

import gdown
import kagglehub
import numpy as np
import pandas as pd

# Download dataset from Kaggle
martininf1n1ty_rna_mutations_all_datasets_path = kagglehub.dataset_download(
    "martininf1n1ty/rna-mutations-all-datasets"
)

print("Data source import complete.")

# Ensure output directory exists
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Download additional parquet file from Google Drive (CCLE prepared file)
gdrive_url = "https://drive.google.com/uc?id=10ey0Mlaibz5kXLVMuEf1kCuaWhS6SwXX"
gdrive_output = data_dir / "filtered_merged_df.parquet"
gdown.download(gdrive_url, str(gdrive_output), quiet=False)
ccle = pd.read_parquet(gdrive_output)
if "symbol_counts" in ccle.columns:
    ccle = ccle.rename(columns={"symbol_counts": "mutation"})
if "primary_site" in ccle.columns:
    ccle["subtype"] = ccle["primary_site"].astype(str)

# Read parquet files
df_expressions = pd.read_parquet(
    os.path.join(
        martininf1n1ty_rna_mutations_all_datasets_path, "expression (1).parquet"
    )
)
df_mutations = pd.read_parquet(
    os.path.join(
        martininf1n1ty_rna_mutations_all_datasets_path, "mutations (1).parquet"
    )
)

# Extract expression gene set from CCLE first row
def _flatten_symbols(value):
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return _flatten_symbols(parsed)
        except (ValueError, SyntaxError):
            return [value]
    if isinstance(value, np.ndarray):
        return _flatten_symbols(value.tolist())
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_symbols(item))
        return flattened
    return [value]


raw_hugo_symbols = ccle.iloc[0]["Hugo_Symbol_expression"]
hugo_symbols_expression = []
for symbol in _flatten_symbols(raw_hugo_symbols):
    if symbol is None:
        continue
    if isinstance(symbol, float) and pd.isna(symbol):
        continue
    symbol_str = str(symbol).strip()
    if symbol_str:
        hugo_symbols_expression.append(symbol_str)

# Keep order but drop duplicates
hugo_symbols_expression = list(dict.fromkeys(hugo_symbols_expression))

# Expression processing:
# 1) filter genes to CCLE set
# 2) build fixed-length expression vectors per case_barcode
df_expressions_filtered = df_expressions[
    df_expressions["gene_name"].isin(hugo_symbols_expression)
]
expression_matrix = (
    df_expressions_filtered.pivot_table(
        index="case_barcode",
        columns="gene_name",
        values="tpm_unstranded",
        aggfunc="mean",
    )
    .reindex(columns=hugo_symbols_expression)
    .fillna(0.0)
)
grouped_expressions_df = pd.DataFrame(
    {
        "case_barcode": expression_matrix.index,
        "tpm_unstranded": expression_matrix.values.tolist(),
    }
).reset_index(drop=True)
filtered_grouped_expressions_df = grouped_expressions_df.copy()

# Mutation processing:
# Count occurrences of Hugo_Symbol per case_barcode and convert row to list
mutation_counts_df = pd.crosstab(
    df_mutations["case_barcode"], df_mutations["Hugo_Symbol"]
).reset_index()
hugo_symbols_mutation = mutation_counts_df.drop("case_barcode", axis=1).columns.tolist()
mutation_data_list = mutation_counts_df.drop("case_barcode", axis=1).values.tolist()
mutation_list_df = pd.DataFrame(
    {
        "case_barcode": mutation_counts_df["case_barcode"],
        "mutation": mutation_data_list,
    }
)
primary_site_df = (
    df_mutations[["case_barcode", "primary_site"]]
    .dropna(subset=["case_barcode"])
    .groupby("case_barcode", as_index=False)["primary_site"]
    .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
)

# Final merge by case_barcode and remove any NaN rows
df_joined = pd.merge(
    filtered_grouped_expressions_df,
    mutation_list_df,
    on="case_barcode",
    how="inner",
).dropna()
df_joined = pd.merge(
    df_joined,
    primary_site_df,
    on="case_barcode",
    how="left",
)
df_joined["subtype"] = df_joined["primary_site"]

# Keep compatibility with previous output naming
df_joined = df_joined.rename(columns={"tpm_unstranded": "expression"})
df_joined["Hugo_Symbol_expression"] = [hugo_symbols_expression] * len(df_joined)
df_joined["Hugo_Symbol_mutation"] = [hugo_symbols_mutation] * len(df_joined)

# Save them into ./data directory
df_expressions_filtered.to_parquet(data_dir / "expression_filtered.parquet")
df_joined.to_parquet(data_dir / "joined.parquet")
ccle.to_parquet(data_dir / "ccle_filtered_merged_df.parquet")

print(f"Zapisano pliki do katalogu: {data_dir.resolve()}")
print(f"Liczba symboli z CCLE do filtra: {len(hugo_symbols_expression)}")
print(f"Liczba wierszy expression po filtrze: {len(df_expressions_filtered)}")
print(f"Liczba case_barcode po grupowaniu: {len(grouped_expressions_df)}")
print(
    "Liczba case_barcode z przygotowanym wektorem ekspresji: "
    f"{len(filtered_grouped_expressions_df)}"
)
print(f"Liczba rekordow po join + dropna: {len(df_joined)}")
print(f"Pobrano plik z Google Drive: {gdrive_output.resolve()}")
PY