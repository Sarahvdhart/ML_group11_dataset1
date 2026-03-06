#importeren van de nodige libraries
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

# Data inlezen
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
X = df.drop(columns=["ID", "label"])

# Nested CV instellen
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # Impute missing values with median, hier kijken of we bijvoorbeeld nog een threshold willen toevoegen voor als er bijvoorbeeld zoveel nullen zijn dat we die feature misschien willen verwijderen
    ('scaler', RobustScaler()), # Scale features using RobustScaler
    ('feature_selection', SelectKBest(score_func=f_classif)), # Select top k features --> ook hier kijken welke feature selection we nog moeten nemen!!!!
    ('classifier', LogisticRegression(max_iter=1000, random_state=42)) # Logistic Regression classifier --> kijken welke classifier we nog moeten nemen!!!!
])

