#!/usr/bin/env python3
"""
Plot line plot with 95% confidence interval band.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib.lines import Line2D


DEFAULT_PALETTE = {
    "Non-Cancer": "#4DBBD5",
    "Cancer": "#E64B35",
}

DEFAULT_BAND_PALETTE = {
    "Non-Cancer": "#DCEAF2",
    "Cancer": "#F9DDDD",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot line plot with confidence interval band."
    )

    parser.add_argument("input_tsv")
    parser.add_argument("output_file")

    parser.add_argument("--x_col", required=True)
    parser.add_argument("--y_col", required=True)
    parser.add_argument("--group_col", default=None)

    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--title", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(
        args.input_tsv,
        sep="\t",
    )

    fig, ax = plt.subplots(
        figsize=(4.8, 3.2)
    )

    if args.group_col is None:
        sns.lineplot(
            data=df,
            x=args.x_col,
            y=args.y_col,
            errorbar=("ci", 95),
            n_boot=1000,
            ax=ax,
        )

    else:
        groups = df[args.group_col].unique()

        for group in groups:
            subset = df[
                df[args.group_col] == group
            ]

            color = DEFAULT_PALETTE.get(
                group,
                None,
            )

            band_color = DEFAULT_BAND_PALETTE.get(
                group,
                None,
            )

            sns.lineplot(
                data=subset,
                x=args.x_col,
                y=args.y_col,
                color=color,
                label=group,
                errorbar=("ci", 95),
                n_boot=1000,
                err_kws={
                    "facecolor": band_color,
                    "alpha": 1,
                    "linewidth": 0,
                    "edgecolor": "none",
                },
                ax=ax,
            )

        if len(groups) == 2:
            mean_df = (
                df.groupby(
                    [args.group_col, args.x_col]
                )[args.y_col]
                .mean()
                .unstack(0)
            )

            if mean_df.shape[1] == 2:
                r_val, _ = spearmanr(
                    mean_df.iloc[:, 0],
                    mean_df.iloc[:, 1],
                    nan_policy="omit",
                )

                handles, labels = ax.get_legend_handles_labels()

                proxy = Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    color="none",
                    label=f"$R={r_val:.3f}$",
                )

                handles.append(proxy)
                labels.append(
                    f"$R={r_val:.3f}$"
                )

                ax.legend(
                    handles=handles,
                    labels=labels,
                    frameon=False,
                    loc="best",
                )

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)

    ax.legend(
        frameon=False,
        loc="best",
    )

    Path(args.output_file).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output_file,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Figure saved to: {args.output_file}")


if __name__ == "__main__":
    main()