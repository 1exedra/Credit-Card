import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df.fillna(0, inplace=True)  

X = df.drop(['Amount', 'Class'], axis=1) 
y = df['Amount']



