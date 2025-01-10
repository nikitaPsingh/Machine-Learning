# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
#print (y)

# Feature scaling
# The standard scaler takes only a 2D array as an input so we need to transform y from 1D to 2D
y = y.reshape(len(y), 1) # y.reshape(no. of rows, no. of cols)
# print(y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
sc_y = StandardScaler() 
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# print(X)
# print(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result

# y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))
# print(y_pred)

y_pred_scaled = regressor.predict(sc_X.transform([[6.5]]))
y_pred_original = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
print(y_pred_original)

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()