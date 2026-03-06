import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import RobustScaler

from worclipo.load_data import load_data


df= load_data()

# 1. Splits de dataframe in de eerste 3 kolommen en de rest
df_to_keep = df.iloc[:, :3]
df_to_scale = df.iloc[:, 3:]

# 2. Initialiseer en fit de scaler
scaler = RobustScaler()
scaled_array = scaler.fit_transform(df_to_scale)

# 3. Maak van de geschaalde array weer een DataFrame
# We geven de kolomnamen en de index weer mee om alles netjes te houden
df_scaled = pd.DataFrame(
    scaled_array, 
    columns=df_to_scale.columns, 
    index=df_to_scale.index)


#Create the total dataframe with a robust scaler applicated
df_final = pd.concat([df_to_keep, df_scaled], axis=1)

#Now we check for outliers in the scaled dataframe. We consider values < -5 or > 5 as extreme outliers.
outlier_masker = (df_scaled < -3) | (df_scaled > 3)

#Count the amount of outliers per column
outliers_per_kolom = outlier_masker.sum()

# Filter the columns that have more than 0 outliers and sort them in ascending order
resultaat = outliers_per_kolom[outliers_per_kolom > 0].sort_values(ascending=True)

# Print the results
print("Aantal extreme uitschieters per kolom (> 3 of < -3):")
print(resultaat)

#Print the amount of columns that has more than 0 outliers
print(f"The amount of columns with outliers: {resultaat.shape[0]}")
#Amount of columns that has more than x% outliers
total_rows = df_scaled.shape[0]
percentage = 0.1
columns_with_outliers = (resultaat > (percentage * total_rows)).sum()
print(f"\nAantal kolommen met meer dan {percentage*100}% uitschieters: {columns_with_outliers}")

