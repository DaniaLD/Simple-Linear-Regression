import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import dataset
dataset = pd.read_csv('./Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting simple linear regression to the TRAINING set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Payment Prediction (Training)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='orange')
plt.title('Payment Prediction (Test)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()