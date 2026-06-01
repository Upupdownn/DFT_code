import numpy as np
from scipy import stats

def compute_midrank(x):
    """
    Compute midranks for a 1D array.

    Midranks are used in DeLong's test to correctly handle tied values.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Input values.

    Returns
    -------
    midranks : ndarray of shape (n_samples,)
        Midrank values (1-based indexing).
    """

    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]

    n = len(sorted_x)
    midranks = np.zeros(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1

        midranks[i:j] = 0.5 * (i + j - 1)
        i = j

    out = np.empty(n, dtype=np.float64)
    out[order] = midranks + 1

    return out


def fast_delong(y_true, score1, score2):
    """
    Perform DeLong's test for comparing two correlated ROC AUCs.

    Reference
    ---------
    DeLong ER, DeLong DM, Clarke-Pearson DL.
    Comparing the areas under two or more correlated ROC curves:
    a nonparametric approach.
    Biometrics. 1988;44(3):837-845.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels.

    score1 : array-like of shape (n_samples,)
        Prediction scores from model 1.

    score2 : array-like of shape (n_samples,)
        Prediction scores from model 2.

    Returns
    -------
    auc1 : float
        ROC AUC of model 1.

    auc2 : float
        ROC AUC of model 2.

    p_value : float
        Two-sided p-value from DeLong's test.
    """

    y_true = np.asarray(y_true).flatten()
    score1 = np.asarray(score1).flatten()
    score2 = np.asarray(score2).flatten()

    if len(np.unique(y_true)) != 2:
        raise ValueError("Only binary classification is supported.")

    labels = np.unique(y_true)
    neg_label, pos_label = labels[0], labels[1]

    pos_idx = np.where(y_true == pos_label)[0]
    neg_idx = np.where(y_true == neg_label)[0]

    m = len(pos_idx)
    n = len(neg_idx)

    if m == 0 or n == 0:
        raise ValueError("Both positive and negative samples are required.")

    def compute_structural_components(score):
        X = score[pos_idx]
        Y = score[neg_idx]

        combined = np.concatenate([X, Y])
        ranks = compute_midrank(combined)

        rank_X = ranks[:m]
        rank_Y = ranks[m:]

        V10 = (rank_X - compute_midrank(X)) / n
        V01 = 1.0 - (rank_Y - compute_midrank(Y)) / m

        auc = np.mean(V10)

        return V10, V01, auc

    def sample_covariance(a, b):
        return np.cov(a, b, ddof=1)[0, 1]

    V10_1, V01_1, auc1 = compute_structural_components(score1)
    V10_2, V01_2, auc2 = compute_structural_components(score2)

    var_1 = sample_covariance(V10_1, V10_1) / m + \
            sample_covariance(V01_1, V01_1) / n

    var_2 = sample_covariance(V10_2, V10_2) / m + \
            sample_covariance(V01_2, V01_2) / n

    cov_12 = sample_covariance(V10_1, V10_2) / m + \
             sample_covariance(V01_1, V01_2) / n

    variance_diff = var_1 + var_2 - 2 * cov_12

    if variance_diff <= 0:
        return auc1, auc2, 1.0

    z_score = (auc1 - auc2) / np.sqrt(variance_diff)
    p_value = 2 * stats.norm.sf(abs(z_score))

    return auc1, auc2, p_value