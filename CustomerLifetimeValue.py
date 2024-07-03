import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df.fillna(0, inplace=True)  

X = df.drop(['Amount', 'Class'], axis=1) 
y = df['Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

