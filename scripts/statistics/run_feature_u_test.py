#!/usr/bin/env python3
"""
Run column-wise Mann-Whitney U tests between cancer and non-cancer samples.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


LABEL_COL = "label"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Mann-Whitney U tests for each feature."
    )

    parser.add_argument("feature_tsv")
    parser.add_argument("info_tsv")
    parser.add_argument("output_tsv")

    return parser.parse_args()


def main():
    args = parse_args()

    feature_df = pd.read_csv(args.feature_tsv, sep="\t", index_col=0)
    info_df = pd.read_csv(args.info_tsv, sep="\t", index_col=0)

    if LABEL_COL not in info_df.columns:
        raise ValueError(f"'{LABEL_COL}' column was not found in info file.")

    common_samples = feature_df.index.intersection(info_df.index)

    if len(common_samples) == 0:
        raise ValueError("No common samples were found between feature and info files.")

    feature_df = feature_df.loc[common_samples]
    labels = info_df.loc[common_samples, LABEL_COL]

    cancer_idx = labels == 1
    non_cancer_idx = labels == 0

    if cancer_idx.sum() == 0 or non_cancer_idx.sum() == 0:
        raise ValueError("Both cancer and non-cancer samples are required.")

    results = []

    for feature in feature_df.columns:
        cancer_values = feature_df.loc[cancer_idx, feature].dropna()
        non_cancer_values = feature_df.loc[non_cancer_idx, feature].dropna()

        if len(cancer_values) == 0 or len(non_cancer_values) == 0:
            p_value = float("nan")
        else:
            _, p_value = mannwhitneyu(
                cancer_values,
                non_cancer_values,
                alternative="two-sided",
            )

        results.append({
            "Feature": feature,
            "P-value": p_value,
        })

    result_df = pd.DataFrame(results)

    valid_mask = result_df["P-value"].notna()

    result_df["FDR"] = float("nan")

    result_df.loc[valid_mask, "FDR"] = multipletests(
        result_df.loc[valid_mask, "P-value"],
        method="fdr_bh",
    )[1]

    result_df = result_df.sort_values(
        by=["FDR", "P-value"],
        ascending=True,
    )

    Path(args.output_tsv).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        args.output_tsv,
        sep="\t",
        index=False,
    )

    print(f"Feature-level U test results saved to: {args.output_tsv}")


if __name__ == "__main__":
    main()