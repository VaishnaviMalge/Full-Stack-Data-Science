import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r"C:\Users\Avinash\Downloads\logit classification.csv")

# split data to dv, iv, test, train

x = dataset.iloc[:,[2,3]].values     # .values - convert dataframe slice to numpy array (required by sklearn)
y= dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=0)

#-----------------------------------------------------------------------------------------------------------------------------------
# standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#-----------------------------------------------------------------------------------------------------------------------------------
# logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

#model-classifier
#algorithm-LogisticRegression

y_pred = classifier.predict(x_test)

#------------------------------------------------------------------------------------------------------------------------------------
# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)

# to get classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr)

# is it good modle
bias = classifier.score(x_train,y_train)
bias

varience = classifier.score(x_test,y_test)
varience

#------------------------------------------------------------------------------------------------------------------------------------
# validation using future data(it have no dependent variable)

dataset1 = pd.read_csv(r"C:\Users\Avinash\Downloads\validation data.csv")

d2 = dataset1.copy()

dataset1 = dataset1.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

M = sc.fit_transform(dataset1)
y_pred = pd.DataFrame()

d2['y_pred'] = classifier.predict(M)

d2.to_csv('final1.csv')
d2

# first we created copy of validation data to preserve it as d2
# selected related coulmns, scaled it and stored in M
# created empty dataframe y_pred
# used scaled data to predict future and stored it as column y_pred in d2
# saved the final d2 file that have original validation data + y_pred