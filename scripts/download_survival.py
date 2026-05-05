#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download TCGA Pan-Cancer survival data from UCSC Xena and align it with
the patients already present in data/joined.parquet.

Source
------
Liu et al. 2018 Pan-Cancer clinical data (UCSC Xena Pan-Can Atlas Hub):
  Survival_SupplementalTable_S1_20171025_xena_sp

Columns in the output file (data/survival.parquet)
---------------------------------------------------
  case_barcode  – 12-char TCGA patient ID  (e.g. TCGA-AA-3525)
  OS            – overall survival event       0 = censored, 1 = deceased
  OS.time       – overall survival time (days)
  DSS           – disease-specific survival event
  DSS.time      – disease-specific survival time (days)
  PFI           – progression-free interval event
  PFI.time      – progression-free interval time (days)
  cancer_type   – primary_site from joined.parquet (where available)

Usage
-----
  python scripts/download_survival.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import certifi
import requests
import pandas as pd

XENA_SURVIVAL_URL = (
    "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/"
    "Survival_SupplementalTable_S1_20171025_xena_sp"
)

SURVIVAL_COLS = ["sample", "OS", "OS.time", "DSS", "DSS.time", "PFI", "PFI.time"]


def _patient_barcode(barcode: str) -> str:
    """Return the 12-char patient-level TCGA barcode (first 3 hyphen-separated fields)."""
    parts = str(barcode).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(barcode)[:12]


def download_xena_survival(url: str) -> pd.DataFrame:
    print(f"Pobieranie danych przezycia z:\n  {url}")
    resp = requests.get(url, timeout=120, verify=certifi.where())
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content), sep="\t", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    present = [c for c in SURVIVAL_COLS if c in df.columns]
    df = df[present].copy()
    df.rename(columns={"sample": "case_barcode"}, inplace=True)
    # Ensure patient-level barcode (Xena already uses 12-char, but normalise)
    df["case_barcode"] = df["case_barcode"].apply(_patient_barcode)
    df = df.drop_duplicates(subset=["case_barcode"])
    print(f"  Xena: {len(df):,} unikalnych pacjentow")
    return df


def load_joined_patients(joined_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(joined_path, columns=["case_barcode", "primary_site"])
    df["case_barcode"] = df["case_barcode"].apply(_patient_barcode)
    df = df.drop_duplicates(subset=["case_barcode"])
    df.rename(columns={"primary_site": "cancer_type"}, inplace=True)
    print(f"  joined.parquet: {len(df):,} unikalnych pacjentow")
    return df


def main(data_dir: Path) -> None:
    joined_path = data_dir / "joined.parquet"
    out_path = data_dir / "survival.parquet"

    if not joined_path.exists():
        raise FileNotFoundError(
            f"Brak pliku {joined_path}. Najpierw uruchom scripts/download_data.sh."
        )

    survival_df = download_xena_survival(XENA_SURVIVAL_URL)
    patients_df = load_joined_patients(joined_path)

    merged = patients_df.merge(survival_df, on="case_barcode", how="left")

    n_total = len(merged)
    n_os = merged["OS.time"].notna().sum()
    print(
        f"\nWyniki laczenia:\n"
        f"  Pacjenci ogolnie    : {n_total:,}\n"
        f"  Z danymi OS         : {n_os:,}  ({n_os / n_total:.1%})\n"
        f"  Bez danych OS       : {n_total - n_os:,}"
    )

    if n_os == 0:
        raise RuntimeError(
            "Zaden pacjent nie zostal dopasowany do danych przezycia.\n"
            "Sprawdz format barcode w joined.parquet."
        )

    # Numeric coercion (Xena sometimes stores as object due to redacted entries)
    for col in ["OS", "OS.time", "DSS", "DSS.time", "PFI", "PFI.time"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged.to_parquet(out_path, index=False)
    print(f"\nZapisano: {out_path.resolve()}")

    print("\nStatystyki OS.time (dni):")
    print(merged["OS.time"].describe().to_string())

    if "cancer_type" in merged.columns:
        print(f"\nTypy nowotworow ({merged['cancer_type'].nunique()} unikatowych):")
        ct_counts = merged["cancer_type"].value_counts()
        print(ct_counts.head(20).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Katalog z plikami danych (domyslnie: data)",
    )
    args = parser.parse_args()
    main(Path(args.data_dir))
