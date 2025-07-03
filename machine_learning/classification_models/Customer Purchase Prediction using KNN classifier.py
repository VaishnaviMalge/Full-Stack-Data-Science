import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Avinash\Downloads\logit classification.csv")

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=0)

#-----------------------------------------------------------------------------------------------------------------------------------
# standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# knn classifier (default k value=5)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=7,weights='distance', p=1 ,algorithm = 'kd_tree' )
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