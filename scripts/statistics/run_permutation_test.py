#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CURRENT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from utils.model_utils import BinaryModel
from utils.frequency_utils import fft_transform


N_SPLITS = 10
N_REPEATS = 10
LABEL_COL = "label"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run permutation test with FFT amplitude features and repeated stratified CV SVM."
    )

    parser.add_argument("cv_feature_tsv")
    parser.add_argument("cv_info_tsv")
    parser.add_argument("output_tsv")

    parser.add_argument("--val_feature_tsv", default=None)
    parser.add_argument("--val_info_tsv", default=None)

    parser.add_argument(
        "--n_permutations",
        type=int,
        default=1000,
        help="Number of permutations. Default: 1000.",
    )

    return parser.parse_args()


def load_xy(feature_file, info_file):
    feature_df = pd.read_csv(feature_file, sep="\t", index_col=0)
    info_df = pd.read_csv(info_file, sep="\t", index_col=0)

    if LABEL_COL not in info_df.columns:
        raise ValueError(f"'{LABEL_COL}' column was not found in info file.")

    common_samples = feature_df.index.intersection(info_df.index)

    if len(common_samples) == 0:
        raise ValueError("No common samples were found between feature and info files.")

    feature_df = feature_df.loc[common_samples]
    y = info_df.loc[common_samples, LABEL_COL].values

    return feature_df, y


def adjust_n_splits(y, n_splits=N_SPLITS):
    _, counts = np.unique(y, return_counts=True)
    max_splits = counts.min()

    if max_splits < 2:
        raise ValueError("Each class must contain at least 2 samples.")

    return min(n_splits, max_splits)


def get_fft_amplitude_features(x):
    amp, _ = fft_transform(
        x,
        axis=1,
        preprocess=True,
        half_spectrum=True,
        remove_dc=True,
    )
    return amp


def repeated_cv_score(x, y_perm, seed):
    n_splits = adjust_n_splits(y_perm)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=N_REPEATS,
        random_state=seed,
    )

    score_sum = np.zeros(len(y_perm), dtype=float)
    score_count = np.zeros(len(y_perm), dtype=float)

    for train_idx, test_idx in cv.split(x, y_perm):
        model = clone(BinaryModel.get_base_model("SVM"))

        model.fit(
            x[train_idx],
            y_perm[train_idx],
        )

        score_sum[test_idx] += model.predict_proba(
            x[test_idx]
        )[:, 1]

        score_count[test_idx] += 1

    return score_sum / score_count


def repeated_val_score(x_cv, y_cv_perm, x_val, seed):
    n_splits = adjust_n_splits(y_cv_perm)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=N_REPEATS,
        random_state=seed,
    )

    val_scores = []

    for train_idx, _ in cv.split(x_cv, y_cv_perm):
        model = clone(BinaryModel.get_base_model("SVM"))

        model.fit(
            x_cv[train_idx],
            y_cv_perm[train_idx],
        )

        val_scores.append(
            model.predict_proba(x_val)[:, 1]
        )

    return np.mean(val_scores, axis=0)


def main():
    args = parse_args()

    has_val = (
        args.val_feature_tsv is not None
        and args.val_info_tsv is not None
    )

    cv_x_df, cv_y = load_xy(
        args.cv_feature_tsv,
        args.cv_info_tsv,
    )

    if has_val:
        val_x_df, val_y = load_xy(
            args.val_feature_tsv,
            args.val_info_tsv,
        )

        common_features = cv_x_df.columns.intersection(val_x_df.columns)

        if len(common_features) == 0:
            raise ValueError("No common features were found between CV and validation matrices.")

        cv_x_df = cv_x_df.loc[:, common_features]
        val_x_df = val_x_df.loc[:, common_features]

        val_x_raw = val_x_df.values
    else:
        val_y = None
        val_x_raw = None

    cv_x_raw = cv_x_df.values

    rng = np.random.default_rng(42)
    results = []

    for i in range(args.n_permutations):
        seed = 42 + i

        row_perm = rng.permutation(cv_x_raw.shape[0])

        cv_x_perm = cv_x_raw[row_perm, :]
        cv_y_perm = cv_y[row_perm]

        cv_x_fft = get_fft_amplitude_features(cv_x_perm)

        cv_score = repeated_cv_score(
            cv_x_fft,
            cv_y_perm,
            seed=seed,
        )

        row = {
            "Permutation": i + 1,
            "CV_AUC": roc_auc_score(cv_y_perm, cv_score),
        }

        if has_val:
            val_row_perm = rng.permutation(val_x_raw.shape[0])

            val_x_perm = val_x_raw[val_row_perm, :]
            val_y_perm = val_y[val_row_perm]

            val_x_fft = get_fft_amplitude_features(val_x_perm)

            val_score = repeated_val_score(
                cv_x_fft,
                cv_y_perm,
                val_x_fft,
                seed=seed,
            )

            row["Validation_AUC"] = roc_auc_score(
                val_y_perm,
                val_score,
            )

        results.append(row)

        if (i + 1) % 10 == 0:
            print(f"Finished {i + 1}/{args.n_permutations} permutations")

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

    print(f"Permutation test results saved to: {args.output_tsv}")


if __name__ == "__main__":
    main()