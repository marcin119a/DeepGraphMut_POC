#!/usr/bin/env python3
"""
Network propagation of gene/DNA profiles.

Supported input formats (--fmt):

  matrix   [default]  Tab-separated, rows=genes, cols=samples:
                         barcode  sample1  sample2
                         GENE1    0.178    0.573
                         GENE2    0.337    0.556

  matrix-T             Tab-separated, rows=samples, cols=genes:
                         barcode  GENE1   GENE2
                         sample1  0.178   0.337
                         sample2  0.573   0.556

  list                 Two-column tab-separated list (sample <TAB> gene):
                         TCGA-A6-2671  TP53
                         TCGA-A6-2671  KRAS
                         TCGA-A6-2672  APC

Output:
  Tab-separated, rows=genes, cols=samples (same as 'matrix' format).

Example (using bundled COAD data):
  python propagate_profile.py \\
      data/ALL/features_mut_list.txt \\
      data/HumanNet90_Symbol.txt \\
      --fmt list -o data/ALL/features_mut_propagated.tsv -v
"""

import argparse
import sys
import numpy as np
import pandas as pd

from pyNBS import data_import_tools as dit
from pyNBS import network_propagation as prop
from pyNBS import pyNBS_core as core


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('profile_file',
        help='Path to input profile file')
    parser.add_argument('network_file',
        help='Path to network edge list file (tab-separated, first two columns = gene pair)')
    parser.add_argument('-o', '--output', default='propagated_profile.tsv',
        help='Output file path (default: propagated_profile.tsv)')
    parser.add_argument('--fmt', choices=['matrix', 'matrix-T', 'list'], default='matrix',
        help='Input file format (default: matrix)')
    parser.add_argument('-a', '--alpha', type=float, default=0.7,
        help='Propagation coefficient, range (0,1) (default: 0.7)')
    parser.add_argument('--no-qnorm', action='store_true',
        help='Skip quantile normalization after propagation')
    parser.add_argument('--symmetric-norm', action='store_true',
        help='Use symmetric adjacency normalization (default: column-sum)')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Print progress messages')
    return parser.parse_args()


def load_profile(path, fmt, verbose):
    """Load profile and return DataFrame with rows=samples, cols=genes."""
    if fmt == 'list':
        # Two-column list: sample <TAB> gene
        f = open(path)
        lines = f.read().splitlines()
        f.close()
        pairs = [(line.split('\t')[0], line.split('\t')[1]) for line in lines if line.strip()]
        idx = pd.MultiIndex.from_tuples(pairs, names=['sample', 'gene'])
        series = pd.Series(1, index=idx)
        sm_mat = series.unstack(fill_value=0)
        if verbose:
            print(f'  Loaded list: {sm_mat.shape[0]} samples x {sm_mat.shape[1]} genes')
        return sm_mat  # rows=samples, cols=genes

    elif fmt == 'matrix':
        # rows=genes, cols=samples → transpose
        df = pd.read_csv(path, sep='\t', index_col=0)
        if verbose:
            print(f'  Loaded matrix: {df.shape[0]} genes x {df.shape[1]} samples')
        return df.T  # rows=samples, cols=genes

    elif fmt == 'matrix-T':
        # rows=samples, cols=genes → already correct
        df = pd.read_csv(path, sep='\t', index_col=0)
        if verbose:
            print(f'  Loaded matrix-T: {df.shape[0]} samples x {df.shape[1]} genes')
        return df  # rows=samples, cols=genes


def main():
    args = parse_args()

    # --- Load profile ---
    if args.verbose:
        print(f'Loading profile ({args.fmt}): {args.profile_file}')
    sm_mat = load_profile(args.profile_file, args.fmt, args.verbose)

    # --- Load network ---
    if args.verbose:
        print(f'Loading network: {args.network_file}')
    network = dit.load_network_file(
        args.network_file, delimiter='\t', verbose=args.verbose
    )
    network_nodes = list(network.nodes)
    if args.verbose:
        print(f'  {len(network_nodes)} nodes, {network.number_of_edges()} edges')

    # --- Check gene overlap ---
    overlap = set(sm_mat.columns).intersection(set(network_nodes))
    if args.verbose:
        print(f'  Genes overlapping network: {len(overlap)} / {len(sm_mat.columns)}')
    if len(overlap) == 0:
        print('ERROR: No genes from profile found in network. Check gene name format.',
              file=sys.stderr)
        sys.exit(1)

    # --- Compute propagation kernel ---
    if args.verbose:
        print(f'Computing propagation kernel (alpha={args.alpha})...')
    network_I = pd.DataFrame(
        np.identity(len(network_nodes)),
        index=network_nodes,
        columns=network_nodes
    )
    kernel = prop.network_propagation(
        network, network_I,
        alpha=args.alpha,
        symmetric_norm=args.symmetric_norm,
        verbose=args.verbose
    )

    # --- Propagate ---
    if args.verbose:
        print('Propagating profile over network...')
    prop_data = prop.network_kernel_propagation(
        network, kernel, sm_mat, verbose=args.verbose
    )

    # --- Quantile normalization ---
    if not args.no_qnorm:
        if args.verbose:
            print('Applying quantile normalization...')
        prop_data = core.qnorm(prop_data)

    # --- Save (transpose: rows=genes, cols=samples) ---
    result = prop_data.T
    result.index.name = 'barcode'
    result.to_csv(args.output, sep='\t')
    print(f'Saved: {args.output}  ({result.shape[0]} genes x {result.shape[1]} samples)')


if __name__ == '__main__':
    main()
