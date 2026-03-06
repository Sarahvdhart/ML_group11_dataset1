import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, nan_threshold=0.30, zero_threshold=0.95, clip_iqr=True):
        self.nan_threshold = nan_threshold
        self.zero_threshold = zero_threshold
        self.clip_iqr = clip_iqr
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # 1) Features met te veel NaN droppen
        nan_ratio = X.isna().mean()
        keep_nan = nan_ratio <= self.nan_threshold

        # 2) Features met te veel nullen droppen
        zero_ratio = (X == 0).mean()
        keep_zero = zero_ratio <= self.zero_threshold

        # 3) Drop features with zero variance 
        variance = X.var()
        keep_var = variance > 0

        # Combine criteria
        self.keep_columns_ = X.columns[keep_nan & keep_zero & keep_var]
        X = X[self.keep_columns_]

        # 3) Mediaan berekenen voor imputatie
        self.medians_ = X.median()

        # Impute op train voor verdere stats
        X = X.fillna(self.medians_)

        # 4) IQR grenzen berekenen (optioneel)
        if self.clip_iqr:
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_ = Q1 - 1.5 * IQR
            self.upper_ = Q3 + 1.5 * IQR

        # 5) scaler fitten
        self.scaler.fit(X)

        return self

    def transform(self, X):
        X = pd.DataFrame(X)

        # dezelfde features houden
        X = X[self.keep_columns_]

        # imputeren
        X = X.fillna(self.medians_)

        # outlier clipping
        if self.clip_iqr:
            X = X.clip(self.lower_, self.upper_, axis=1)

        # scaling
        X_scaled = self.scaler.transform(X)

        return X_scaled