#!/usr/bin/env python3
"""
Run repeated stratified cross-validation models and save prediction scores.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CURRENT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from utils.model_utils import BinaryModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated stratified CV models."
    )

    parser.add_argument("cv_feature_tsv")
    parser.add_argument("cv_info_tsv")
    parser.add_argument("cv_score_tsv")

    parser.add_argument("--val_feature_tsv", default=None)
    parser.add_argument("--val_info_tsv", default=None)
    parser.add_argument("--val_score_tsv", default=None)

    parser.add_argument(
        "--models",
        nargs="+",
        default=["SVM"],
        choices=["SVM", "LR", "RF", "GBDT"],
        help="Models to run. Default: SVM.",
    )

    parser.add_argument(
        "--nested-cv",
        action="store_true",
        help="Use nested CV with GridSearchCV.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    has_val = (
        args.val_feature_tsv is not None
        and args.val_info_tsv is not None
        and args.val_score_tsv is not None
    )

    cv_score_list = []
    val_score_list = []

    for method in args.models:
        print(f"\nRunning model: {method}")

        model = BinaryModel(
            method=method,
            cv_feature=args.cv_feature_tsv,
            cv_info=args.cv_info_tsv,
            val_feature=args.val_feature_tsv if has_val else None,
            val_info=args.val_info_tsv if has_val else None,
            nested_cv=args.nested_cv,
        )

        model.fit_cv()

        cv_score_df = model.get_cv_score_df()
        cv_score_df = cv_score_df.rename(
            columns={"score": method}
        )
        cv_score_list.append(cv_score_df)

        print(f"{method} cv AUC: {model.cv_auc:.4f}")

        if has_val:
            model.validate()

            val_score_df = model.get_val_score_df()
            val_score_df = val_score_df.rename(
                columns={"score": method}
            )
            val_score_list.append(val_score_df)

            print(f"{method} validation AUC: {model.val_auc:.4f}")

    cv_out_df = pd.concat(cv_score_list, axis=1)
    cv_out_df.index.name = "sample"

    Path(args.cv_score_tsv).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    cv_out_df.to_csv(args.cv_score_tsv, sep="\t")

    print(f"\nCV scores saved to: {args.cv_score_tsv}")

    if has_val:
        val_out_df = pd.concat(val_score_list, axis=1)
        val_out_df.index.name = "sample"

        Path(args.val_score_tsv).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        val_out_df.to_csv(args.val_score_tsv, sep="\t")

        print(f"Validation scores saved to: {args.val_score_tsv}")


if __name__ == "__main__":
    main()