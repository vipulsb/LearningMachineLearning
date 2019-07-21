#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:21:19 2019

@author: vipulbhandare
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('Data.csv')

#Handle missing elements
#independant columns
x = dataset.iloc[:, :-1].values
#dependant column
y = dataset.iloc[:, 3].values

#missing elements
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
x = oneHotEncoder.fit_transform(x).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)