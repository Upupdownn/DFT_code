#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an SVM model and predict scores on validation set."
    )

    parser.add_argument("train_feature_tsv")
    parser.add_argument("train_info_tsv")
    parser.add_argument("val_feature_tsv")
    parser.add_argument("val_info_tsv")
    parser.add_argument("output_score_tsv")

    parser.add_argument(
        "--norm",
        action="store_true",
        help="Apply column-wise standardization before training.",
    )

    parser.add_argument(
        "--pca",
        action="store_true",
        help="Apply PCA retaining 95%% explained variance.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_x = pd.read_csv(args.train_feature_tsv, sep="\t", index_col=0)
    train_info = pd.read_csv(args.train_info_tsv, sep="\t", index_col=0)

    val_x = pd.read_csv(args.val_feature_tsv, sep="\t", index_col=0)
    val_info = pd.read_csv(args.val_info_tsv, sep="\t", index_col=0)

    train_samples = train_x.index.intersection(train_info.index)
    val_samples = val_x.index.intersection(val_info.index)

    train_x = train_x.loc[train_samples]
    train_y = train_info.loc[train_samples, "label"]

    val_x = val_x.loc[val_samples]

    common_features = train_x.columns.intersection(val_x.columns)

    train_x = train_x.loc[:, common_features]
    val_x = val_x.loc[:, common_features]

    steps = []

    if args.norm:
        steps.append(("scaler", StandardScaler()))

    if args.pca:
        steps.append(("pca", PCA(n_components=0.95)))

    steps.append(("svm", SVC(probability=True, random_state=42)))

    model = Pipeline(steps)

    model.fit(train_x, train_y)

    train_score = model.predict_proba(train_x)[:, 1]
    val_score = model.predict_proba(val_x)[:, 1]

    train_auc = roc_auc_score(train_y, train_score)
    val_auc = roc_auc_score(val_info.loc[val_samples, "label"], val_score)

    print(f"Train AUC: {train_auc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    output_df = pd.DataFrame({"score": val_score}, index=val_x.index)

    output_df.index.name = "sample"

    Path(args.output_score_tsv).parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(args.output_score_tsv, sep="\t")


if __name__ == "__main__":
    main()