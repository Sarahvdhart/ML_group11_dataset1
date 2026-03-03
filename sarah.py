#importeren
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from worclipo.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')


print(data.head())

#aantal label
aantal_labels = data["label"].count()
print(aantal_labels)

#print names van labels
names_label = set(data["label"])
print(names_label)

#statistieken per groep (count, mean, variance, std dev)
count_per_groep = data.groupby("label").count()
mean_per_groep = data.groupby("label").mean()
variance_per_groep = data.groupby("label").var()
std_dev_per_groep = data.groupby("label").std()
print("count_per_groep:")
print(count_per_groep)
print("mean_per_groep:")    
print(mean_per_groep)
print("variance_per_groep:")
print(variance_per_groep)
print("std_dev_per_groep:")
print(std_dev_per_groep)

# #variantie per feature

# Alleen numerieke kolommen selecteren
numeric_data = data.select_dtypes(include="number")

# Variantie berekenen
variance_per_feature = numeric_data.var()

# Aantal features met variantie 0 tellen
zero_variance_count = (variance_per_feature == 0).sum()

print("Aantal features met variantie 0:", zero_variance_count)

# Optioneel: welke features hebben variantie 0?
zero_variance_features = variance_per_feature[variance_per_feature == 0]
print("\nFeatures met variantie 0:")
print(zero_variance_features)

#robust scaler 
from sklearn.preprocessing import RobustScaler
# Alle numerieke kolommen selecteren
numeric_cols = data.select_dtypes(include="number").columns 
# RobustScaler toepassen op de numerieke kolommen
scaler = RobustScaler()
data_scaled = data.copy()
data_scaled[numeric_cols] = scaler.fit_transform(data[numeric_cols])