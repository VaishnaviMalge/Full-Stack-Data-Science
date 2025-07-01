import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r"C:\Users\Avinash\Downloads\logit classification.csv")

# split data to dv, iv, test, train

x = dataset.iloc[:,[2,3]].values
y= dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=0)

#-------------------------------------------------------
# standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

#model-classifier
#algorithm-LogisticRegression

y_pred = classifier.predict(x_test)

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


# build model 20% split ratio,normalization,random state= 11, 21, 51, 101
