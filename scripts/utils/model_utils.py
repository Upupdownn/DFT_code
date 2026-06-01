import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score


class BinaryModel:
    """
    Binary classification model for repeated stratified CV and optional
    nested cross-validation using GridSearchCV.
    """

    def __init__(
        self,
        method="SVM",
        cv_feature=None,
        cv_info=None,
        val_feature=None,
        val_info=None,
        label_col="label",
        n_splits=10,
        n_repeats=10,
        nested_cv=False,
        random_state=42,
    ):
        self.method = method
        self.cv_feature = cv_feature
        self.cv_info = cv_info
        self.val_feature = val_feature
        self.val_info = val_info
        self.label_col = label_col
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.nested_cv = nested_cv
        self.random_state = random_state

        self.models = []
        self.cv_score = None
        self.val_score = None
        self.cv_auc = None
        self.val_auc = None

        self.x_cv, self.y_cv = self._load_xy(cv_feature, cv_info)

        if val_feature is not None and val_info is not None:
            self.x_val, self.y_val = self._load_xy(val_feature, val_info)
            common_features = self.x_cv.columns.intersection(self.x_val.columns)
            self.x_cv = self.x_cv.loc[:, common_features]
            self.x_val = self.x_val.loc[:, common_features]
        else:
            self.x_val, self.y_val = None, None

    def _load_xy(self, feature_file, info_file):
        feature_df = pd.read_csv(feature_file, sep="\t", index_col=0)
        info_df = pd.read_csv(info_file, sep="\t", index_col=0)

        if self.label_col not in info_df.columns:
            raise ValueError(f"'{self.label_col}' column was not found in info file.")

        common_samples = feature_df.index.intersection(info_df.index)

        if len(common_samples) == 0:
            raise ValueError("No common samples were found between feature and info files.")

        x = feature_df.loc[common_samples]
        y = info_df.loc[common_samples, self.label_col].values

        return x, y

    @staticmethod
    def get_base_model(method: str):
        method = method.upper()

        if method == "SVM":
            return SVC(probability=True, random_state=42)

        if method == "LR":
            return LogisticRegression(random_state=42)

        if method == "RF":
            return RandomForestClassifier(n_jobs=-1)

        if method in ["GBDT", "GBM"]:
            return GradientBoostingClassifier(random_state=42)

        raise ValueError("method must be one of: SVM, LR, RF, GBDT")

    @staticmethod
    def get_param_grid(method: str):
        method = method.upper()
        if method == "SVM-Linear":
            return {'C': np.logspace(-3, 3, 7), 'kernel': ['linear']}
        
        if method == ["SVM", "SVM-Grid"]:
            return {
                "C": np.logspace(-3, 3, 7),
                "gamma": np.logspace(-3, 3, 7),
                "kernel": ["rbf", "linear"],
            }

        if method == "LR":
            return {
                "C": np.logspace(-3, 3, 7),
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
            }

        if method == "RF":
            return {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            }

        if method in ["GBDT", "GBM"]:
            return {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [2, 3, 5],
            }

        raise ValueError("method must be one of: SVM/SVM-Linear/SVM-Grid, LR, RF, GBDT/GBM")

    def _build_model(self, seed):
        base_model = self.get_base_model(self.method)

        if not self.nested_cv:
            return clone(base_model)

        inner_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        return GridSearchCV(
            estimator=base_model,
            param_grid=self.get_param_grid(self.method),
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
        )

    def fit_cv(self):
        """
        Train repeated stratified CV models and generate out-of-fold scores
        for the cv set.
        """
        x = self.x_cv.values
        y = self.y_cv

        class_counts = np.bincount(y)
        max_splits = np.min(class_counts)

        n_splits = min(
            self.n_splits,
            max_splits
        )

        if n_splits < 2:
            raise ValueError("Each class must contain at least 2 samples for cross-validation.")

        if n_splits != self.n_splits:
            print(
                f"Warning: n_splits adjusted from "
                f"{self.n_splits} to {n_splits} "
                f"because of limited class size."
            )

        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        score_sum = np.zeros(len(y), dtype=float)
        score_count = np.zeros(len(y), dtype=float)

        self.models = []

        for fold_id, (cv_idx, test_idx) in enumerate(cv.split(x, y)):
            repeat_idx = fold_id // self.n_splits + 1
            fold_idx = fold_id % self.n_splits + 1
            if fold_idx == 1: print(f"Repeat {repeat_idx}/{self.n_repeats}: ", end='')
            print(f"Fold {fold_idx}/{self.n_splits} ", end='')
            if fold_idx == self.n_splits: print()

            model = self._build_model(seed=fold_id)

            model.fit(x[cv_idx], y[cv_idx])

            score = model.predict_proba(x[test_idx])[:, 1]

            score_sum[test_idx] += score
            score_count[test_idx] += 1

            self.models.append(model)

        self.cv_score = score_sum / score_count
        self.cv_auc = roc_auc_score(y, self.cv_score)

        return self

    def validate(self):
        """
        Predict validation scores by averaging predictions from all CV models.
        """
        if self.x_val is None or self.y_val is None:
            raise ValueError("Validation feature and info files were not provided.")

        if len(self.models) == 0:
            raise ValueError("No model found. Run fit_cv() first.")

        x_val = self.x_val.values

        scores = []
        for model in self.models:
            scores.append(model.predict_proba(x_val)[:, 1])

        self.val_score = np.mean(scores, axis=0)
        self.val_auc = roc_auc_score(self.y_val, self.val_score)

        return self

    def fit_full(self):
        """
        Train one final model using the full cv set.
        """
        model = self._build_model(seed=self.random_state)
        model.fit(self.x_cv.values, self.y_cv)

        self.full_model = model
        self.full_cv_score = model.predict_proba(self.x_cv.values)[:, 1]
        self.full_cv_auc = roc_auc_score(self.y_cv, self.full_cv_score)

        if self.x_val is not None:
            self.full_val_score = model.predict_proba(self.x_val.values)[:, 1]
            self.full_val_auc = roc_auc_score(self.y_val, self.full_val_score)

        return self

    def get_cv_score_df(self):
        return pd.DataFrame(
            {"score": self.cv_score},
            index=self.x_cv.index,
        ).rename_axis("sample")

    def get_val_score_df(self):
        return pd.DataFrame(
            {"score": self.val_score},
            index=self.x_val.index,
        ).rename_axis("sample")