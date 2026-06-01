#!/usr/bin/env python3

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

# Allow importing from scripts/utils/
CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CURRENT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from utils.statistics_utils import fast_delong


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pairwise DeLong tests for prediction scores."
    )

    parser.add_argument(
        "--score1file",
        required=True,
        help="Path to the first score file. Rows are samples and columns are models.",
    )

    parser.add_argument(
        "--score2file",
        required=True,
        help="Path to the second score file. Rows are samples and columns are models.",
    )

    parser.add_argument(
        "--info_file",
        required=True,
        help="Path to the sample information file containing a 'label' column.",
    )

    parser.add_argument(
        "--output",
        default="delong_results.tsv",
        help="Output path for DeLong test results. Default: delong_results.tsv",
    )

    parser.add_argument(
        "--sep",
        default="\t",
        help="Column separator for input and output files. Default: tab.",
    )

    parser.add_argument(
        "--label_col",
        default="label",
        help="Column name of the binary label in info_file. Default: label.",
    )

    return parser.parse_args()


def read_table(path, sep):
    return pd.read_csv(path, sep=sep, index_col=0)


def main():
    args = parse_args()

    score1_df = read_table(args.score1file, args.sep)
    score2_df = read_table(args.score2file, args.sep)
    info_df = read_table(args.info_file, args.sep)

    if args.label_col not in info_df.columns:
        raise ValueError(f"'{args.label_col}' column was not found in info_file.")

    common_samples = score1_df.index.intersection(score2_df.index).intersection(info_df.index)

    if len(common_samples) == 0:
        raise ValueError("No common samples were found among score1file, score2file, and info_file.")

    score1_df = score1_df.loc[common_samples]
    score2_df = score2_df.loc[common_samples]
    info_df = info_df.loc[common_samples]

    y_true = info_df[args.label_col].values             # [0, 0, ..., 1, 1]

    merged_scores = pd.concat([score1_df, score2_df], axis=1)

    if merged_scores.columns.duplicated().any():
        raise ValueError(
            "Duplicated model names were found after merging score1file and score2file. "
            "Please make model column names unique."
        )

    results = []

    for model1, model2 in itertools.combinations(merged_scores.columns, 2):
        score1 = merged_scores[model1].values
        score2 = merged_scores[model2].values

        auc1, auc2, p_value = fast_delong(y_true, score1, score2)

        results.append({
            "Model_1": model1,
            "Model_2": model2,
            "AUC_1": auc1,
            "AUC_2": auc2,
            "Delta_AUC": auc1 - auc2,
            "DeLong_P": p_value,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, sep=args.sep, index=False)

    print(f"DeLong test completed.")
    print(f"Number of samples: {len(common_samples)}")
    print(f"Number of models: {merged_scores.shape[1]}")
    print(f"Number of pairwise comparisons: {len(result_df)}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()