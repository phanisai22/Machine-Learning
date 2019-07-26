# Importing libraries.
from sklearn.preprocessing import Imputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset.
data_frame = pd.read_csv("Data.csv")
x = data_frame.iloc[:, :-1].values
y = data_frame.iloc[:, 3].values

# Taking care of missing values.
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)