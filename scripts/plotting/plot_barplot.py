#!/usr/bin/env python3
"""
Plot grouped barplot from a TSV file.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_COLORS = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot grouped barplot from TSV file."
    )

    parser.add_argument("input_tsv")
    parser.add_argument("output_file")

    parser.add_argument("--x_col", required=True)
    parser.add_argument("--y_col", required=True)
    parser.add_argument("--group_col", required=True)

    parser.add_argument("--ci_low_col", default=None)
    parser.add_argument("--ci_high_col", default=None)

    parser.add_argument("--title", default=None)
    parser.add_argument("--ylabel", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(
        args.input_tsv,
        sep="\t",
    )

    fig, ax = plt.subplots(
        figsize=(6, 3.5)
    )

    unique_groups = list(df[args.group_col].unique())
    order = list(df[args.x_col].unique())

    palette = {
        g: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        for i, g in enumerate(unique_groups)
    }

    sns.barplot(
        data=df,
        x=args.x_col,
        y=args.y_col,
        hue=args.group_col,
        palette=palette,
        order=order,
        errorbar=None,
        edgecolor=None,
        ax=ax,
    )

    if (
        args.ci_low_col is not None
        and args.ci_high_col is not None
    ):
        df_plot = df.copy()

        df_plot[args.x_col] = pd.Categorical(
            df_plot[args.x_col],
            categories=order,
            ordered=True,
        )

        df_plot[args.group_col] = pd.Categorical(
            df_plot[args.group_col],
            categories=unique_groups,
            ordered=True,
        )

        df_plot = df_plot.sort_values(
            by=[args.group_col, args.x_col]
        )

        for patch, (_, row) in zip(
            ax.patches,
            df_plot.iterrows(),
        ):
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()

            ci_low = row[args.ci_low_col]
            ci_high = row[args.ci_high_col]

            lower_err = max(0, y - ci_low)
            upper_err = max(0, ci_high - y)

            ax.errorbar(
                x=x,
                y=y,
                yerr=[[lower_err], [upper_err]],
                fmt="none",
                ecolor="#4D4D4D",
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                zorder=10,
            )

    ax.set_xlabel(None)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    ax.yaxis.grid(
        True,
        linestyle="--",
        color="#E6E6E6",
    )

    ax.set_axisbelow(True)

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

    print(f"Barplot saved to: {args.output_file}")


if __name__ == "__main__":
    main()