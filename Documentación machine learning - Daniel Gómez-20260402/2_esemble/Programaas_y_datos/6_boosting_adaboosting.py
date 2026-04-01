# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:16:40 2023

@author: dagom
"""

# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Import Support Vector Classifier
from sklearn.svm import SVC

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,  
                         learning_rate=1, random_state=1234)

abc_svm = AdaBoostClassifier(n_estimators=50, base_estimator = SVC(probability=True), 
                         learning_rate=1, random_state=1234)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

model2 = abc_svm.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model2.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#############