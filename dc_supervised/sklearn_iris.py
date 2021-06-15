from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(type(iris))
print(iris.data.shape)

X = iris.data
Y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())
df['target'] = iris.target
print(df.head(6))
