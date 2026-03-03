import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score # Voor het uitvoeren van cross-validatie
from sklearn.pipeline import Pipeline # Voor het maken van een machine learning pipeline
from sklearn.preprocessing import RobustScaler # Voor het schalen van de features, vooral handig bij outliers
from sklearn.linear_model import LogisticRegression # Voor het trainen van een logistieke regressie model
from sklearn.model_selection import GridSearchCV # Voor het uitvoeren van een grid search om de beste hyperparameters te vinden
from sklearn.feature_selection import SelectKBest, f_classif # Voor het selecteren van de beste features op basis van ANOVA F-test

df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")

print(df.shape)   # zou iets als (115, 494) moeten geven
print(df.isnull().sum().sum())

# Labels en features scheiden
y = df["label"]
X = df.drop(columns=["ID", "label"])

# Labels omzetten naar 0/1
y = y.map({"lipoma": 0, "liposarcoma": 1}) # Zorg ervoor dat de labels correct worden omgezet

print("X shape:", X.shape) 
print("Class distribution:\n", y.value_counts()) # Controleer of de klassen in balans zijn

# Nested CV splits
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Voor de buitenste loop van de nested CV, waarbij we de data in 5 splits verdelen
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1) # Voor de binnenste loop van de nested CV, waarbij we de data in 5 splits verdelen voor hyperparameter tuning

print("Totaal:", len(y))
print("Totaal class balans:", y.value_counts().to_dict(), "\n")

# outer folds
for i, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y), start=1):
    y_outer_train = y.iloc[outer_train_idx]
    y_outer_test  = y.iloc[outer_test_idx]

    print(f"OUTER fold {i}")
    print("  outer train size:", len(outer_train_idx), " outer test size:", len(outer_test_idx))
    print("  outer train classes:", y_outer_train.value_counts().to_dict())
    print("  outer test  classes:", y_outer_test.value_counts().to_dict())
    print("-" * 50)

#inner folds
outer_train_idx, outer_test_idx = next(outer_cv.split(X, y))

X_outer_train = X.iloc[outer_train_idx]
y_outer_train = y.iloc[outer_train_idx]

print("Outer-train size:", len(y_outer_train))
print("Outer-train classes:", y_outer_train.value_counts().to_dict(), "\n")

for j, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train), start=1):
    y_inner_train = y_outer_train.iloc[inner_train_idx]
    y_inner_val   = y_outer_train.iloc[inner_val_idx]

    print(f"  INNER fold {j}")
    print("    inner train size:", len(inner_train_idx), " inner val size:", len(inner_val_idx))
    print("    inner train classes:", y_inner_train.value_counts().to_dict())
    print("    inner val   classes:", y_inner_val.value_counts().to_dict())

# voorbeeld: outer fold 1 train/test sets
X_train = X.iloc[outer_train_idx] 
X_test  = X.iloc[outer_test_idx]
y_train = y.iloc[outer_train_idx]
y_test  = y.iloc[outer_test_idx]

print(X_train.shape, X_test.shape)
print(y_train.value_counts(), "\n", y_test.value_counts())

