import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt


boston = pd.read_csv('boston.csv')
print(boston.head())

X = boston.drop("MEDV", axis=1).values
y = boston['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42)


reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(reg.score(X_test, y_test))

plt.scatter(X_train, y_train)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show()


