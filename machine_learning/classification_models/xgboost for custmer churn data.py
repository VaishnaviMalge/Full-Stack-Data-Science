import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r"E:\Vaishnavi\dataset\Churn_Modelling.csv")

x = dataset.iloc[:,3:-1].values         # consider only relevant attribute
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] =le.fit_transform(x[:,2])
print(x)

# to stop data leakage when data have more than one categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

# feature scalingnot required in decision tree

# train test

from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

from xgboost import XGBClassifier
classfier = XGBClassifier(random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)


bias = classifier.score(x_train,y_train)
print(bias)

varience = classifier.score(x_test,y_test)
print(varience)

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
