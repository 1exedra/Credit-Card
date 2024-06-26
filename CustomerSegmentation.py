import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('creditcard.csv')

data.fillna(method='ffill', inplace=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

