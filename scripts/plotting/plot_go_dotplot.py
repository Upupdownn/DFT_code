#!/usr/bin/env python3
"""
Plot GO enrichment dotplot.

Required input columns:
    Description, GeneRatio, p.adjust, Count
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GO enrichment dotplot."
    )

    parser.add_argument("go_tsv")
    parser.add_argument("output_file")

    return parser.parse_args()


def parse_gene_ratio(x):
    if isinstance(x, str) and "/" in x:
        numerator, denominator = x.split("/")
        return float(numerator) / float(denominator)

    return float(x)


def main():
    args = parse_args()

    go_df = pd.read_csv(args.go_tsv, sep="\t")

    required_cols = ["Description", "GeneRatio", "p.adjust", "Count"]
    missing_cols = [c for c in required_cols if c not in go_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    go_df = go_df.loc[:, required_cols].copy()

    go_df["GeneRatio"] = go_df["GeneRatio"].apply(parse_gene_ratio)

    go_df = (
        go_df
        .sort_values("p.adjust", ascending=True)
        .head(20)
        .sort_values("GeneRatio", ascending=True)
    )

    go_df["Description_wrap"] = go_df["Description"].apply(
        lambda x: "\n".join(textwrap.wrap(str(x), width=25))
    )

    min_dot_size = 20
    max_dot_size = 60

    min_count = go_df["Count"].min()
    max_count = go_df["Count"].max()

    if max_count == min_count:
        go_df["DotSize"] = (min_dot_size + max_dot_size) / 2
    else:
        go_df["DotSize"] = (
            (go_df["Count"] - min_count)
            / (max_count - min_count)
            * (max_dot_size - min_dot_size)
            + min_dot_size
        )

    fig, ax = plt.subplots(
        figsize=(6, 0.4 * len(go_df))
    )

    scatter = ax.scatter(
        x=go_df["GeneRatio"],
        y=go_df["Description_wrap"],
        s=go_df["DotSize"],
        c=-np.log10(go_df["p.adjust"]),
        cmap="RdBu_r",
        marker="o",
    )

    cax = inset_axes(
        ax,
        width="8%",
        height="30%",
        loc="lower left",
        bbox_to_anchor=(0.82, 0.03, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(scatter, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.set_title(
        r"$-\log_{10}(\mathrm{FDR})$",
        rotation=0,
        fontweight="normal",
        fontsize=8,
    )

    if max_count == min_count:
        legend_counts = [int(min_count)]
    else:
        legend_counts = [
            int(min_count),
            int((min_count + max_count) / 2),
            int(max_count),
        ]

    for count in legend_counts:
        if max_count == min_count:
            size = (min_dot_size + max_dot_size) / 2
        else:
            size = (
                (count - min_count)
                / (max_count - min_count)
                * (max_dot_size - min_dot_size)
                + min_dot_size
            )

        ax.scatter(
            [],
            [],
            s=size,
            c="gray",
            label=str(count),
            marker="o",
        )

    ax.legend(
        scatterpoints=1,
        frameon=False,
        title="Count",
        labelspacing=1,
        loc="lower left",
        bbox_to_anchor=(0.35, 0.03),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Gene Ratio")
    ax.set_ylabel(None)
    ax.set_title("Enriched GO Term\n(Biological Process)")

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

    print(f"GO dotplot saved to: {args.output_file}")


if __name__ == "__main__":
    main()