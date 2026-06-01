#!/usr/bin/env python3
"""
Plot ROC curves from one or more score files.

Each score file can contain multiple model score columns.
Each info file must contain a 'label' column.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CURRENT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from utils.metrics_utils import calc_auc_ci
from utils.plot_utils import set_publication_style


LABEL_COL = "label"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ROC curves from multiple score and info files."
    )

    parser.add_argument(
        "--score_files",
        nargs="+",
        required=True,
        help="Score TSV files. Rows are samples and columns are model scores.",
    )

    parser.add_argument(
        "--info_files",
        nargs="+",
        required=True,
        help="Info TSV files. Must match score_files one-to-one.",
    )

    parser.add_argument(
        "output_file",
        help="Output figure file, e.g. roc.pdf, roc.svg, or roc.png.",
    )

    return parser.parse_args()


def load_score_label(score_file, info_file):
    score_df = pd.read_csv(score_file, sep="\t", index_col=0)
    info_df = pd.read_csv(info_file, sep="\t", index_col=0)

    if LABEL_COL not in info_df.columns:
        raise ValueError(f"'{LABEL_COL}' column was not found in info file: {info_file}")

    common_samples = score_df.index.intersection(info_df.index)

    if len(common_samples) == 0:
        raise ValueError(
            f"No common samples were found between {score_file} and {info_file}"
        )

    score_df = score_df.loc[common_samples]
    y_true = info_df.loc[common_samples, LABEL_COL].values

    return score_df, y_true


def make_curve_name(score_file, model_name, n_models):
    dataset_name = Path(score_file).stem

    if n_models == 1:
        return dataset_name

    return f"{dataset_name}_{model_name}"


def main():
    args = parse_args()

    if len(args.score_files) != len(args.info_files):
        raise ValueError("The number of score_files must match the number of info_files.")

    set_publication_style(scale=1.0)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    for score_file, info_file in zip(args.score_files, args.info_files):
        score_df, y_true = load_score_label(score_file, info_file)

        for model_name in score_df.columns:
            y_score = score_df[model_name].values

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_value = roc_auc_score(y_true, y_score)

            ci_res = calc_auc_ci(
                y_true,
                y_score,
                n_bootstrap=1000,
                alpha=0.95,
                seed=42,
            )

            curve_name = make_curve_name(
                score_file,
                model_name,
                n_models=score_df.shape[1],
            )

            label = (
                f"{curve_name}: "
                f"{auc_value:.3f} "
                f"({ci_res['AUC_95CI_lower']:.3f}-{ci_res['AUC_95CI_upper']:.3f})"
            )

            ax.plot(
                fpr,
                tpr,
                label=label,
            )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=0.8,
        color="#D9D9D9",
        zorder=-10,
    )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC Curve")

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    Path(args.output_file).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(args.output_file, bbox_inches="tight")
    plt.close(fig)

    print(f"ROC figure saved to: {args.output_file}")


if __name__ == "__main__":
    main()