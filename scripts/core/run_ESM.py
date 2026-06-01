#!/usr/bin/env python3
"""
Run Ensemble Spectrum Model (ESM).

For each feature scale file:
    1. Train four base models: SVM, LR, RF, GBDT.
    2. Generate CV scores for cross-validation samples.
    3. Generate averaged validation scores if validation data are provided.

Then:
    4. Use all base-model scores as new features.
    5. Train a second-level SVM using repeated stratified CV.
    6. Save final cv and validation scores.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CURRENT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from utils.model_utils import BinaryModel


BASE_MODELS = ["SVM", "LR", "RF", "GBDT"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ESM using multi-scale feature directories."
    )

    parser.add_argument("cv_feature_dir")
    parser.add_argument("cv_info_tsv")
    parser.add_argument("cv_score_tsv")

    parser.add_argument("--val_feature_dir", default=None)
    parser.add_argument("--val_info_tsv", default=None)
    parser.add_argument("--val_score_tsv", default=None)

    return parser.parse_args()


def get_tsv_files(feature_dir):
    feature_dir = Path(feature_dir)
    files = sorted(feature_dir.glob("*.tsv"))

    if len(files) == 0:
        raise ValueError(f"No TSV files found in: {feature_dir}")

    return files


def check_matching_files(cv_files, val_files):
    cv_names = [f.name for f in cv_files]
    val_names = [f.name for f in val_files]

    if cv_names != val_names:
        raise ValueError(
            "Train and validation feature directories must contain "
            "the same TSV files with the same names."
        )


def main():
    args = parse_args()

    has_val = (
        args.val_feature_dir is not None
        and args.val_info_tsv is not None
        and args.val_score_tsv is not None
    )

    cv_files = get_tsv_files(args.cv_feature_dir)

    if has_val:
        val_files = get_tsv_files(args.val_feature_dir)
        check_matching_files(cv_files, val_files)
    else:
        val_files = [None] * len(cv_files)

    esm_cv_features = []
    esm_val_features = []

    for cv_file, val_file in zip(cv_files, val_files):
        scale_name = cv_file.stem

        print(f"\nProcessing scale: {scale_name}")

        for method in BASE_MODELS:
            print(f"Base model: {method}")

            model = BinaryModel(
                method=method,
                cv_feature=str(cv_file),
                cv_info=args.cv_info_tsv,
                val_feature=str(val_file) if has_val else None,
                val_info=args.val_info_tsv if has_val else None,
                nested_cv=False,
            )

            model.fit_cv()

            cv_score_df = model.get_cv_score_df().rename(
                columns={"score": f"{scale_name}_{method}"}
            )
            esm_cv_features.append(cv_score_df)

            print(f"{scale_name}_{method} cv AUC: {model.cv_auc:.4f}")

            if has_val:
                model.validate()

                val_score_df = model.get_val_score_df().rename(
                    columns={"score": f"{scale_name}_{method}"}
                )
                esm_val_features.append(val_score_df)

                print(f"{scale_name}_{method} validation AUC: {model.val_auc:.4f}")

    esm_cv_df = pd.concat(esm_cv_features, axis=1)
    esm_cv_df.index.name = "sample"

    temp_cv_feature = Path(args.cv_score_tsv).with_suffix(".esm_features.cv.tmp.tsv")
    temp_cv_feature.parent.mkdir(parents=True, exist_ok=True)
    esm_cv_df.to_csv(temp_cv_feature, sep="\t")

    if has_val:
        esm_val_df = pd.concat(esm_val_features, axis=1)
        esm_val_df.index.name = "sample"

        temp_val_feature = Path(args.val_score_tsv).with_suffix(".esm_features.val.tmp.tsv")
        temp_val_feature.parent.mkdir(parents=True, exist_ok=True)
        esm_val_df.to_csv(temp_val_feature, sep="\t")
    else:
        temp_val_feature = None

    print("\nTraining second-level ESM SVM")

    esm_model = BinaryModel(
        method="SVM",
        cv_feature=str(temp_cv_feature),
        cv_info=args.cv_info_tsv,
        val_feature=str(temp_val_feature) if has_val else None,
        val_info=args.val_info_tsv if has_val else None,
        nested_cv=False,
    )

    esm_model.fit_cv()

    final_cv_score = esm_model.get_cv_score_df().rename(
        columns={"score": "ESM"}
    )

    Path(args.cv_score_tsv).parent.mkdir(parents=True, exist_ok=True)
    final_cv_score.to_csv(args.cv_score_tsv, sep="\t")

    print(f"ESM cv AUC: {esm_model.cv_auc:.4f}")
    print(f"Final cv scores saved to: {args.cv_score_tsv}")

    if has_val:
        esm_model.validate()

        final_val_score = esm_model.get_val_score_df().rename(
            columns={"score": "ESM"}
        )

        Path(args.val_score_tsv).parent.mkdir(parents=True, exist_ok=True)
        final_val_score.to_csv(args.val_score_tsv, sep="\t")

        print(f"ESM validation AUC: {esm_model.val_auc:.4f}")
        print(f"Final validation scores saved to: {args.val_score_tsv}")
    
    if temp_cv_feature.exists():
        temp_cv_feature.unlink()

    if has_val and temp_val_feature.exists():
        temp_val_feature.unlink()


if __name__ == "__main__":
    main()