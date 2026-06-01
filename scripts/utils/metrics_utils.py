import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def calc_mds(x, normalize=True):
    """
    Calculate Motif Diversity Score (MDS) for each sample.

    MDS is defined as the normalized Shannon entropy
    computed row-wise from the feature matrix.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Input feature matrix.

    normalize : bool, default=True
        Whether to normalize entropy by log(n_features).

    Returns
    -------
    ndarray of shape (n_samples,)
        Row-wise MDS values.
    """

    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    row_sums = x.sum(axis=1, keepdims=True)
    probs = np.divide(
        x,
        row_sums,
        out=np.zeros_like(x),
        where=row_sums != 0
    )

    mds = -np.sum(
        probs * np.log(
            probs,
            out=np.zeros_like(probs),
            where=probs > 1e-12
        ),
        axis=1
    )

    if normalize:
        mds /= np.log(probs.shape[1])

    return mds


def calc_auc(y_true, y_score):
    """
    Calculate ROC AUC.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_score : array-like
        Prediction scores.

    Returns
    -------
    float
        ROC AUC value.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    return roc_auc_score(y_true, y_score)


def calc_sensitivity_at_specificity(y_true, y_score, specificity=0.95):
    """
    Calculate sensitivity at a given specificity.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_score : array-like
        Prediction scores.
    specificity : float, default=0.95
        Target specificity.

    Returns
    -------
    float
        Sensitivity at the target specificity.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    specificity_values = 1.0 - fpr

    valid = specificity_values >= specificity
    if not np.any(valid):
        return np.nan

    return np.max(tpr[valid])


def calc_auc_ci(y_true, y_score, n_bootstrap=1000, alpha=0.95, seed=42):
    """
    Calculate ROC AUC and its bootstrap confidence interval.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    auc_values = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        y_true_bs = y_true[indices]
        y_score_bs = y_score[indices]

        if len(np.unique(y_true_bs)) < 2:
            continue

        auc_values.append(roc_auc_score(y_true_bs, y_score_bs))

    auc_values = np.asarray(auc_values)

    lower_q = (1.0 - alpha) / 2.0 * 100
    upper_q = (1.0 + alpha) / 2.0 * 100

    return {
        "AUC": roc_auc_score(y_true, y_score),
        f"AUC_{int(alpha * 100)}CI_lower": np.percentile(auc_values, lower_q),
        f"AUC_{int(alpha * 100)}CI_upper": np.percentile(auc_values, upper_q),
    }


def calc_auc_sensitivity_ci(
    y_true,
    y_score,
    n_bootstrap=1000,
    alpha=0.95,
    seed=42,
    specificities=(0.98, 0.95, 0.85),
):
    """
    Calculate ROC AUC, sensitivity at fixed specificities,
    and their bootstrap confidence intervals.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    auc_values = []
    sensitivity_values = {sp: [] for sp in specificities}

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        y_true_bs = y_true[indices]
        y_score_bs = y_score[indices]

        if len(np.unique(y_true_bs)) < 2:
            continue

        auc_values.append(roc_auc_score(y_true_bs, y_score_bs))

        for sp in specificities:
            sensitivity_values[sp].append(
                calc_sensitivity_at_specificity(y_true_bs, y_score_bs, specificity=sp)
            )

    lower_q = (1.0 - alpha) / 2.0 * 100
    upper_q = (1.0 + alpha) / 2.0 * 100

    auc_values = np.asarray(auc_values)

    result = {
        "AUC": roc_auc_score(y_true, y_score),
        f"AUC_{int(alpha * 100)}CI_lower": np.percentile(auc_values, lower_q),
        f"AUC_{int(alpha * 100)}CI_upper": np.percentile(auc_values, upper_q),
    }

    for sp in specificities:
        sp_label = int(sp * 100)
        sens_arr = np.asarray(sensitivity_values[sp])
        sens_arr = sens_arr[~np.isnan(sens_arr)]

        result[f"Sensitivity_at_{sp_label}Specificity"] = calc_sensitivity_at_specificity(
            y_true, y_score, specificity=sp
        )

        if len(sens_arr) == 0:
            result[f"Sensitivity_at_{sp_label}Specificity_{int(alpha * 100)}CI_lower"] = np.nan
            result[f"Sensitivity_at_{sp_label}Specificity_{int(alpha * 100)}CI_upper"] = np.nan
        else:
            result[f"Sensitivity_at_{sp_label}Specificity_{int(alpha * 100)}CI_lower"] = np.percentile(
                sens_arr, lower_q
            )
            result[f"Sensitivity_at_{sp_label}Specificity_{int(alpha * 100)}CI_upper"] = np.percentile(
                sens_arr, upper_q
            )

    return result


def calc_auc_sensitivity_ci_stratified(
    y_true,
    y_score,
    n_bootstrap=1000,
    alpha=0.95,
    seed=42,
    specificities=(0.98, 0.95, 0.85),
):
    """
    Calculate ROC AUC, sensitivity at fixed specificities,
    and confidence intervals using stratified bootstrap sampling.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("y_true must contain both positive and negative samples.")

    rng = np.random.default_rng(seed)

    auc_values = []
    sensitivity_values = {sp: [] for sp in specificities}

    for _ in range(n_bootstrap):
        bs_pos_idx = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bs_neg_idx = rng.choice(neg_idx, size=len(neg_idx), replace=True)

        bs_idx = np.concatenate([bs_pos_idx, bs_neg_idx])
        rng.shuffle(bs_idx)

        y_true_bs = y_true[bs_idx]
        y_score_bs = y_score[bs_idx]

        auc_values.append(roc_auc_score(y_true_bs, y_score_bs))

        for sp in specificities:
            sensitivity_values[sp].append(
                calc_sensitivity_at_specificity(y_true_bs, y_score_bs, specificity=sp)
            )

    lower_q = (1.0 - alpha) / 2.0 * 100
    upper_q = (1.0 + alpha) / 2.0 * 100

    auc_values = np.asarray(auc_values)

    result = {
        "AUC": roc_auc_score(y_true, y_score),
        f"AUC_{int(alpha * 100)}CI_lower": np.percentile(auc_values, lower_q),
        f"AUC_{int(alpha * 100)}CI_upper": np.percentile(auc_values, upper_q),
    }

    for sp in specificities:
        sp_label = int(sp * 100)
        sens_arr = np.asarray(sensitivity_values[sp])
        sens_arr = sens_arr[~np.isnan(sens_arr)]

        result[f"Sensitivity_at_{sp_label}Specificity"] = calc_sensitivity_at_specificity(
            y_true, y_score, specificity=sp
        )

        if len(sens_arr) == 0:
            result[f"Sensitivity_at_{sp_label}Specificity_{int(alpha * 100)}CI_lower"] = np.nan
            result[f"Sensitivity_at_{sp_label}Specificity_{int(alpha * 100)}CI_upper"] = np.nan
        else:
            result[f"Sensitivity_at_{sp_label}Specificity_{int(alpha * 100)}CI_lower"] = np.percentile(
                sens_arr, lower_q
            )
            result[f"Sensitivity_at_{sp_label}Specificity_{int(alpha * 100)}CI_upper"] = np.percentile(
                sens_arr, upper_q
            )

    return result