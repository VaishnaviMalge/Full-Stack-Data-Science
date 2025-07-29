import numpy as np
import pandas as pd
import Matplotlib.pyplot as plt
dataset = pd.read_csv(r"C:\Users\Avinash\Downloads\Restaurant_Reviews.tsv", delimiter='\t')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# TfDEf
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(x_train,y_train)
print(bias)
variance = classifier.score(x_test,y_test)
print(variance)
# underfitting
# if we use ml in nlp we get underfit and we can't even add new attribute for training
# we can do data duplication. duplicate data make x times of original data and then build a model you will get desired accuracy