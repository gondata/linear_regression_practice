# First of all we have to import the libraries that we are going to use

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

yf.pdr_override()

# Then we have to download the data

tickers = ['IWM', '^GSPC']
startdate = '2020-01-01'
enddate = '2023-03-01'

data = pdr.get_data_yahoo(tickers, start=startdate, end=enddate)['Adj Close']

# Returns

returns = data.pct_change()

# Variables

X = np.array(returns['IWM'].iloc[1:]).reshape(-1, 1)
y = np.array(returns['^GSPC'].iloc[1:]).reshape(-1, 1)

# Model

model = LinearRegression()  # We create an instance for the lineal regression model
model.fit(X, y) # We adjust the model to the train data

# Prediction

y_pred = model.predict(X)

# Beta

beta = model.coef_  # Slope

# R2

r2 = r2_score(X, y_pred)

# Intercept

intercept = model.intercept_

# Prints

print("Beta: ", beta)
print("R^2: ", r2)
print("Intercept: ", intercept)

# Graph

fig, ax = plt.subplots()
ax.set_title(str(tickers) + "Lineal Regression")
ax.scatter(X, y)
ax.plot(X, y_pred, c="red")
plt.show()