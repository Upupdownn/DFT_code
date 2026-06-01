#!/usr/bin/env python3
"""
Calculate Motif Diversity Score (MDS) from an EDM feature matrix.

Input:
    TSV file with samples as rows and features as columns.

Output:
    TSV file with two columns: sample, MDS
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CURRENT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from utils.metrics_utils import calc_mds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Motif Diversity Score (MDS) from a feature matrix."
    )

    parser.add_argument(
        "input_file",
        help="Input TSV file. Rows are samples and columns are features.",
    )

    parser.add_argument(
        "output_file",
        help="Output TSV file with columns: sample, MDS.",
    )

    parser.add_argument(
        "--sep",
        default="\t",
        help="Column separator. Default: tab.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    feature_df = pd.read_csv(args.input_file, sep=args.sep, index_col=0)

    mds_values = calc_mds(feature_df.values)

    output_df = pd.DataFrame({
        "sample": feature_df.index,
        "MDS": mds_values,
    })

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_file, sep=args.sep, index=False)


if __name__ == "__main__":
    main()