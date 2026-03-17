import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from worclipo.load_data import load_data 
from sklearn.pipeline import Pipeline
from preprocessing import CustomPreprocessor


def get_rf_pipeline():
    return Pipeline([
        ("preprocess", CustomPreprocessor(
            zero_threshold=0.90,
            clip_iqr=False,
            corr_threshold=0.85
        )),
        ("classifier", RandomForestClassifier(random_state=42))
    ])


df = load_data()
df_to_keep = df.iloc[:, :3]
df_to_scale = df.iloc[:, 3:]

scaler = RobustScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_to_scale), 
    columns=df_to_scale.columns, 
    index=df_to_scale.index)

df = pd.concat([df_to_keep, df_scaled], axis=1)

# Splitting the data into features and target variable
X = df.drop('label', axis=1)  # Alle kolommen behalve de target
y = df['label']               # Alleen de kolom die je wilt voorspellen

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
# Initialise
classifier = sk.ensemble.RandomForestClassifier(random_state=42)

random_search = sk.model_selection.RandomizedSearchCV(estimator= classifier,
                                                       param_distributions = param_grid, 
                                                         n_iter=20, 
                                                         random_state= 42, 
                                                         cv=5,
)
random_search.fit(X_train, y_train)

# 1. Zet alle resultaten om in een overzichtelijke tabel
results_df = pd.DataFrame(random_search.cv_results_)

# 2. Sorteer op de hoogste score zodat de beste bovenaan staan
results_df = results_df.sort_values(by='rank_test_score')

# 3. Laat alleen de belangrijkste kolommen zien (parameters en score)
# We filteren de kolommen die beginnen met 'param_' en de 'mean_test_score'
important_cols = [col for col in results_df.columns if col.startswith('param_')] + ['mean_test_score']
print(results_df[important_cols].head(10))


