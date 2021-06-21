import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt


boston = pd.read_csv('boston.csv')
print(boston.head())

X = boston.drop("MEDV", axis=1).values
y = boston['MEDV'].values.reshape(-1, 1)

X_rooms = X[:, 5].reshape(-1, 1)
reg = LinearRegression()
reg.fit(X_rooms, y)

plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show()
