from worclipo.load_data import load_data
import matplotlib.pyplot as plt
import scipy.stats as stats

data = load_data()

print(f'The number of columns: {len(data.columns)}')

data = data.loc[:, data.nunique() > 1] # Niet nodig als we toch die threshold erna doen want deze hebben 0

print(f'The number of columns: {len(data.columns)}') 

data_numeric = data.select_dtypes(include=["number"])
var = data_numeric.var()
selected_columns = var[var > 0.0001].index

data = data[selected_columns]
print(f'The number of columns: {len(data.columns)}')



