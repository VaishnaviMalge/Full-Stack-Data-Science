import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Avinash\Downloads\emp_sal.csv")


# dv and iv

x = dataset.iloc[:,1:2].values         # used column slicing
y = dataset.iloc[:,2].values

# train test not required cause it's small data

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y,color='red')                   # original values
plt.plot(x, lin_reg.predict(x),color='blue')   # predicted values
plt.title("linear rgression model")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

y_pred = lin_reg.predict([[6.5]])
y_pred

# this graph shows high error between actual and predicted value, hence not a good model
# this data is nonlinear needs to use ploymodel

# ----------------------------------------------------------------------------------------------------------------------------------------

# poly model with degree = (default=2)(parameter tunning)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()

x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title("polynomial rgression model")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred


# poly model with degree = 5 hyperparameter tunning

from sklearn.preprocessing import PolynomialFeatures
poly_reg_5 = PolynomialFeatures(degree=5)

x_poly_5 = poly_reg_5.fit_transform(x)

poly_reg_5.fit(x_poly_5,y)

lin_reg_5 = LinearRegression()
lin_reg_5.fit(x_poly_5,y)

plt.scatter(x,y,color='red')
plt.plot(x, lin_reg_5.predict(poly_reg_5.fit_transform(x)),color='blue')
plt.title("polynomial rgression model with degree = 5")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

poly_model_pred_5 = lin_reg_5.predict(poly_reg_5.fit_transform([[6.5]]))
poly_model_pred_5


# by the degree we can get model with more accuracy  -- 
# hyperparameter tunning degree = 5 ; parameter tunning degree = default value = 2
# use tab to get whole suggested code string printed
# linear: y = mx + c
# multilinear: y = m1x1 + m2x2 +...+ mnxn + c
# polynomial: y = m1(x1)^1 + m2(x2)^2 + m3(x3)^3 + m4(x4)^4 +..... mn(xn)^n + c
# diferent variable names are used to understand that this time model is with deifferent degree, else evry other step is same
# polynomial is just to set degrees, after that use linearregression
#------------------------------------------------------------------------------------------------------------------------------------------

# svr(support vector regressor)
from sklearn.svm import SVR
svr_model = SVR(kernel="poly", degree=4) 
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
svr_model_pred

#------------------------------------------------------------------------------------------------------------------------------------------

# knn(k nearest neighbor)
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor()
knn_model.fit(x,y)
 
knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred)

#  ----------------------------------------------------------------------------------------------------------------------------------------

# decision tree ml regression model
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(x,y)

dt_model.predict = dt_model.predict([[6.5]])
dt_model.predict

# -----------------------------------------------------------------------------------------------------------------------------------------

# Random Forest Model 
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=0)             #  random_state - to set seed for reproducibility
rf_model.fit(x,y)

rf_model_pred = rf_model.predict([[6.5]])
rf_model_pred

