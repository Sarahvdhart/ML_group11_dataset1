# lasso model proberen

# Algemene packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Data importeren
from worclipo.load_data import load_data
data = load_data()

# Labels handmatig coderen: lipoma=0, liposarcoma=1
label_order = ["lipoma", "liposarcoma"]
le = LabelEncoder()
le.fit(label_order)
y = le.transform(data["label"])

# Alleen numerieke kolommen selecteren
numeric_data = data.select_dtypes(include="number")

# 1. Features met variantie 0 verwijderen 
selector = VarianceThreshold(threshold=0.0)
numeric_data_var = selector.fit_transform(numeric_data)
feature_names = numeric_data.columns[selector.get_support()]

print(f"Original number of features: {numeric_data.shape[1]}")
print(f"Number of features after removing zero variance: {numeric_data_var.shape[1]}")

# 2️. RobustScaler toepassen
scaler = RobustScaler()
X_scaled = scaler.fit_transform(numeric_data_var)

# # 3️. PCA toepassen **alleen voor visualisatie**
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # PCA scatterplot
# plt.figure(figsize=(8,6))
# sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="Set1")
# plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
# plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
# plt.title("PCA scatterplot van dataset (2 componenten)")
# plt.show()

# 4️.Feature selection op originele geschaalde features
# a) Lasso (L1 regularisatie)
lasso = Lasso(alpha=0.01)  # alpha = sterkte van regularisatie, hoger = meer features weg
lasso.fit(X_scaled, y)
lasso_coef = pd.Series(lasso.coef_, index=feature_names)
selected_features_lasso = lasso_coef[lasso_coef != 0].index.tolist()
print("Belangrijkste features volgens Lasso (L1):")
#aantal selected features
print(f"Aantal geselecteerde features: {len(selected_features_lasso)}")
print(selected_features_lasso)

# b) Random Forest (model-based feature importance)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
rf_importances = pd.Series(rf.feature_importances_, index=feature_names)
selected_features_rf = rf_importances[rf_importances > 0.01].sort_values(ascending=False).index.tolist()
print("Belangrijkste features volgens Random Forest:")
#aantal selected features
print(f"Aantal geselecteerde features: {len(selected_features_rf)}")
print(selected_features_rf)

#visualiseren van feature importances van random forest
import matplotlib.pyplot as plt
import seaborn as sns

# Selecteer de belangrijkste features en hun importances
rf_top_importances = rf_importances[selected_features_rf].sort_values(ascending=True)

# Plot
plt.figure(figsize=(10, max(5, len(rf_top_importances)*0.3)))  # automatisch hoogte aanpassen aan aantal features
sns.barplot(x=rf_top_importances.values, y=rf_top_importances.index, palette="viridis")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest: Belangrijkste features")
plt.tight_layout()
plt.show()
