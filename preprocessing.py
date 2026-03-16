# This file contains the custom preprocessing class used in the pipeline. It includes:

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

# Make a class: CustomPreprocessor, which inherits from BaseEstimator and TransformerMixin, so it can be used in a sklearn pipeline. 
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, nan_threshold=0.30, zero_threshold=0.95, clip_iqr=True): 
        self.nan_threshold = nan_threshold # Threshold for dropping features based on NaN ratio at 30%
        self.zero_threshold = zero_threshold # Threshold for dropping features based on zero ratio at 95%
        self.clip_iqr = clip_iqr # Whether to perform outlier clipping based on IQR
        self.scaler = RobustScaler() # RobustScaler to handle outliers

    # Fit method: calculates which features to keep, the medians for imputation, and the IQR bounds if clipping is enabled. It also fits the scaler on the training data.
    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # 1) Drop features with too many NaNs
        nan_ratio = X.isna().mean()
        keep_nan = nan_ratio <= self.nan_threshold

        # 2) Drop features with too many zeros
        zero_ratio = (X == 0).mean()
        keep_zero = zero_ratio <= self.zero_threshold

        # 3) Drop features with zero variance 
        variance = X.var()
        keep_var = variance > 0

        # 4) Combine criteria: only keep features that pass all three checks
        self.keep_columns_ = X.columns[keep_nan & keep_zero & keep_var]
        X = X[self.keep_columns_]

        # 5) Compute the medians for imputation (after dropping features)
        self.medians_ = X.median()

        # Impute NaNs with medians (this is necessary before fitting the scaler and calculating IQR bounds)
        X = X.fillna(self.medians_)

        # 6) IQR clipping bounds calculation (if enabled)
        if self.clip_iqr:
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_ = Q1 - 1.5 * IQR # Lower bound, check if we will use this later
            self.upper_ = Q3 + 1.5 * IQR # Upper bound, check if we will use this later

        # 7) Fit the scaler
        self.scaler.fit(X)

        return self

    # Transform method: applies the same transformations to the data: keeps the same features, imputes NaNs, performs outlier clipping if enabled, and scales the features using the fitted scaler.
    def transform(self, X):
        X = pd.DataFrame(X)

        # Keep only the columns that were selected during fitting
        X = X[self.keep_columns_]

        # Impute NaNs with the medians calculated during fitting
        X = X.fillna(self.medians_)

        # Outlier clipping based on IQR bounds (if enabled)
        if self.clip_iqr:
            X = X.clip(self.lower_, self.upper_, axis=1)

        # Scale the features using the fitted scaler
        X_scaled = self.scaler.transform(X)

        return X_scaled