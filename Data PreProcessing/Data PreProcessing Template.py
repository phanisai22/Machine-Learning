# Importing libraries.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

# Encoding categorical data.
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])
one_hot_encoder = OneHotEncoder(categorical_features=[0])
x = one_hot_encoder.fit_transform(x).toarray()

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
