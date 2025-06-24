# We have a clent who have invested in different field in different cities. 
# We need to suggest him in which field he should invest further to get a profit

# Import libraries and dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"E:\Vaishnavi\practiced\machine learning\regression\multilinear regression model\Investment.csv")

# seperate dependant and independant variables

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]

# Dummy variable Encoding(category-->int)

x = pd.get_dummies(x, dtype = int)

# sepearate train and test data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# import LinearRegression model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# predict y 

y_pred = regressor.predict(x_test)

# calculating slope and constant

m = regressor.coef_
print(m)

intercept = regressor.intercept_
print(intercept)

# creating a constant column

x = np.append(arr = np.ones((50,1)).astype(int),values = x, axis = 1)


# Backward Elimination. RFE(Recursive Feature Elimination)
# Check p-value if p>0.05 reject null hypothesis /eliminate column.
# enven though dataframe have 7 column we took only 6, cause in one hot encoder if one column value can be guessed by the value of other columns


import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]]
# ordinary least squares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,5]]
# ordinary least squares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3]]
# ordinary least squares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,3]]
# ordinary least squares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1]]
# ordinary least squares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

# check R^2 for train and test data

bias = regressor.score(x_train,y_train)
bias

variance = regressor.score(x_test,y_test)
variance

#  x1(Digital Marketing is the column with least p-value after a backword elimination. So, Digital Marketing is goo field to invest money as it will lead to profit.
# r square for both train and test data are are high --> It is a good model