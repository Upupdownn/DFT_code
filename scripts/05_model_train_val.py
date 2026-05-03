#!/usr/bin/env python3
"""
Train and validate RF, LR, SVM, GBDT models on features.
Output TSV columns: id, RF, LR, SVM, GBDT
"""

import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib
from pathlib import Path
import os


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and validate RF, LR, SVM, GBDT models on features."
    )
    parser.add_argument("feature_file", help="Input TSV file with sample features")
    parser.add_argument("label_file", help="Label TSV file: sample_id, label")
    parser.add_argument("output_file", help="Output TSV file for prediction scores")
    parser.add_argument(
        "--mode",
        choices=["train", "validate"],
        default="train",
        help="Mode: train or validate"
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path to model dir, save for train and load for validate"
    )
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("--n_repeat", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def load_data(feature_file, label_file):
    features = pd.read_table(feature_file, header=0, index_col=0)
    labels = pd.read_table(label_file, header=0, index_col=0)

    common_ids = features.index.intersection(labels.index)
    if len(common_ids) == 0:
        raise ValueError("No common sample IDs between features and labels")

    X = features.loc[common_ids].values
    y = labels.loc[common_ids].iloc[:, 0].to_numpy()
    sample_ids = common_ids

    logger.info(f"Loaded {len(common_ids)} samples with {X.shape[1]} features")
    return X, y, sample_ids


def build_models(random_state=42):
    """All models use default parameters as much as possible."""
    return {
        "RF": RandomForestClassifier(random_state=random_state),
        "LR": LogisticRegression(random_state=random_state, max_iter=1000),
        "SVM": SVC(probability=True, random_state=random_state),
        "GBDT": GradientBoostingClassifier(random_state=random_state),
    }


def cross_validation(X, y, n_fold=5, n_repeat=10, random_state=42):
    all_scores = {}
    all_models = {}

    for model_name in ["RF", "LR", "SVM", "GBDT"]:
        logger.info(f"Training {model_name} with cross-validation...")

        repeat_probs = np.zeros((n_repeat, len(y)), dtype=np.float64)
        model_list = []

        for i in range(n_repeat):
            kfold = StratifiedKFold(
                n_splits=n_fold,
                shuffle=True,
                random_state=random_state + i
            )

            for j, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
                x_train = X[train_idx]
                y_train = y[train_idx]
                x_test = X[test_idx]

                model = build_models(random_state=random_state + i)[model_name]
                model.fit(x_train, y_train)

                repeat_probs[i][test_idx] = model.predict_proba(x_test)[:, 1]
                model_list.append(model)

        y_prob = np.mean(repeat_probs, axis=0)
        auc = roc_auc_score(y, y_prob)

        logger.info(f"{model_name} CV AUC: {auc:.4f}")

        all_scores[model_name] = y_prob
        all_models[model_name] = model_list

    return all_scores, all_models


def validate_model(model_dict, X, y):
    all_scores = {}

    for model_name, model_list in model_dict.items():
        logger.info(f"Validating {model_name}...")

        model_probs = []
        for model in model_list:
            model_probs.append(model.predict_proba(X)[:, 1])

        y_prob = np.mean(model_probs, axis=0)
        auc = roc_auc_score(y, y_prob)

        logger.info(f"{model_name} validate AUC: {auc:.4f}")
        all_scores[model_name] = y_prob

    return all_scores


def save_models(model_dict, model_dir):
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    for model_name, model_list in model_dict.items():
        sub_dir = os.path.join(model_dir, model_name)
        Path(sub_dir).mkdir(parents=True, exist_ok=True)

        for i, model in enumerate(model_list):
            joblib.dump(model, os.path.join(sub_dir, f"{model_name}_model_{i}.pkl"))

    logger.info(f"Models saved in {model_dir}")


def load_models(model_dir):
    model_dict = {}

    for model_name in ["RF", "LR", "SVM", "GBDT"]:
        sub_dir = os.path.join(model_dir, model_name)
        if not os.path.isdir(sub_dir):
            raise ValueError(f"Model directory not found: {sub_dir}")

        model_list = []
        for fname in sorted(os.listdir(sub_dir)):
            if fname.endswith(".pkl"):
                model = joblib.load(os.path.join(sub_dir, fname))
                model_list.append(model)

        if len(model_list) == 0:
            raise ValueError(f"No model files found in {sub_dir}")

        model_dict[model_name] = model_list
        logger.info(f"Loaded {len(model_list)} {model_name} models")

    return model_dict


def save_scores(output_file, sample_ids, scores):
    df = pd.DataFrame({
        "id": sample_ids,
        "RF": scores["RF"],
        "LR": scores["LR"],
        "SVM": scores["SVM"],
        "GBDT": scores["GBDT"],
    })

    df.to_csv(output_file, header=True, index=False, sep="\t")
    logger.info(f"Prediction scores saved in {output_file}")


def main():
    args = parse_args()

    logger.info("Loading data...")
    X, y, sample_ids = load_data(args.feature_file, args.label_file)

    if args.mode == "train":
        logger.info(
            f"Train mode: {args.n_fold}-fold CV, repeat {args.n_repeat} times"
        )

        scores, model_dict = cross_validation(
            X,
            y,
            n_fold=args.n_fold,
            n_repeat=args.n_repeat,
            random_state=args.random_state
        )

        if args.model_dir is not None:
            save_models(model_dict, args.model_dir)

        save_scores(args.output_file, sample_ids, scores)

    elif args.mode == "validate":
        if not args.model_dir:
            raise ValueError("--model_dir required for validate mode")

        logger.info(f"Validate mode: loading models from {args.model_dir}")

        model_dict = load_models(args.model_dir)
        scores = validate_model(model_dict, X, y)

        save_scores(args.output_file, sample_ids, scores)


if __name__ == "__main__":
    main()