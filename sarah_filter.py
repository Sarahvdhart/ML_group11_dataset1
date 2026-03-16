#filtering proberen

#importeren
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from worclipo.load_data import load_data
import numpy as np
from sklearn.feature_selection import chi2

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

data = load_data()

# =====================================================
# 1 Variantie 0 features verwijderen
# =====================================================

numeric_data = data.select_dtypes(include="number")
print(f'Aantal features voor filtering: {len(numeric_data.columns)}')

variance_per_feature = numeric_data.var()

zero_variance_features = variance_per_feature[variance_per_feature == 0].index

print("\nAantal features met variantie 0:", len(zero_variance_features))

# verwijderen
data = data.drop(columns=zero_variance_features)

print("Aantal features na variance filter:", data.shape[1])


# =====================================================
# 2 Robust scaling
# =====================================================

numeric_cols = data.select_dtypes(include="number").columns

scaler = RobustScaler()

data_scaled = data.copy()
data_scaled[numeric_cols] = scaler.fit_transform(data[numeric_cols])


# =====================================================
# 3 Correlation filter (>0.9)
# =====================================================

X = data_scaled.drop(columns=["label"])
y = data_scaled["label"]

corr_matrix = X.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_features = [col for col in upper.columns if any(upper[col] > 0.9)]

print("\nAantal features verwijderd door correlatie:", len(high_corr_features))

X = X.drop(columns=high_corr_features)

print("Aantal features na correlation filter:", X.shape[1])


# =====================================================
# 4 Chi-squared test (p < 0.05)
# =====================================================

# chi2 kan niet met negatieve waarden
X_chi = X.copy()

# shift data zodat alles >=0 is
X_chi = X_chi - X_chi.min()

chi_scores, p_values = chi2(X_chi, y)

p_values = pd.Series(p_values, index=X.columns)

selected_features = p_values[p_values < 0.05].index

removed_chi = len(X.columns) - len(selected_features)

print("\nAantal features verwijderd door Chi²:", removed_chi)

X = X[selected_features]

print("Aantal features na Chi²:", X.shape[1])


# =====================================================
# Eindresultaat
# =====================================================

print("\n====== Eindresultaat ======")
print("Aantal features over:", X.shape[1])
