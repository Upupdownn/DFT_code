#!/usr/bin/env python3
"""
Summarize model scores by AUC, confidence interval, and sensitivity at 95% specificity.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CURRENT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from utils.metrics_utils import (
    calc_auc_sensitivity_ci,
    calc_auc_sensitivity_ci_stratified,
)


LABEL_COL = "label"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize model scores with AUC and sensitivity at 95% specificity."
    )

    parser.add_argument("score_tsv")
    parser.add_argument("info_tsv")
    parser.add_argument("output_tsv")

    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified bootstrap.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    score_df = pd.read_csv(args.score_tsv, sep="\t", index_col=0)
    info_df = pd.read_csv(args.info_tsv, sep="\t", index_col=0)

    if LABEL_COL not in info_df.columns:
        raise ValueError(f"'{LABEL_COL}' column was not found in info file.")

    common_samples = score_df.index.intersection(info_df.index)

    if len(common_samples) == 0:
        raise ValueError("No common samples were found between score and info files.")

    score_df = score_df.loc[common_samples]
    y_true = info_df.loc[common_samples, LABEL_COL].values

    summary_func = (
        calc_auc_sensitivity_ci_stratified
        if args.stratified
        else calc_auc_sensitivity_ci
    )

    results = []

    for model_name in score_df.columns:
        y_score = score_df[model_name].values

        res = summary_func(
            y_true,
            y_score,
            specificities=(0.95,),
        )

        results.append({
            "Model": model_name,
            "AUC": res["AUC"],
            "AUC_95CI_lower": res["AUC_95CI_lower"],
            "AUC_95CI_upper": res["AUC_95CI_upper"],
            "Sensitivity_at_95Specificity": res["Sensitivity_at_95Specificity"],
            "Sensitivity_at_95Specificity_95CI_lower": res[
                "Sensitivity_at_95Specificity_95CI_lower"
            ],
            "Sensitivity_at_95Specificity_95CI_upper": res[
                "Sensitivity_at_95Specificity_95CI_upper"
            ],
        })

    result_df = pd.DataFrame(results)

    Path(args.output_tsv).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        args.output_tsv,
        sep="\t",
        index=False,
    )

    print(f"Score summary saved to: {args.output_tsv}")


if __name__ == "__main__":
    main()