import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('creditcard.csv')

data.fillna(method='ffill', inplace=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

import matplotlib.pyplot as plt
import seaborn as sns
print(data.describe())

for column in data.columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
