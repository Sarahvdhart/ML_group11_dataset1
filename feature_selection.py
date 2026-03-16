
from sklearn.feature_selection import SelectFdr, f_classif


def get_feature_selector(alpha=0.1):
    """
    Feature selection using ANOVA F-test with FDR correction.
    Keeps features with statistically significant association
    with the target variable.
    """
    return SelectFdr(score_func=f_classif, alpha=alpha)