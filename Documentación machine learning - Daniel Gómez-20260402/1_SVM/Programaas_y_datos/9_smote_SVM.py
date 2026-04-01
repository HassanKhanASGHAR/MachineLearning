# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 00:34:36 2023

@author: dagom
"""
# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import string
import re

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# ==============================================================================
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification
X, y = make_classification(n_classes=2, class_sep=1.5,
weights=[0.01, 0.99], n_informative=2, n_redundant=0, flip_y=0,
n_features=2, n_clusters_per_class=2, n_samples=1000, random_state=10)



########### hacemos clasificacion con SVM ####
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.30, random_state = 101)
### DIBUJAMOS
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X[:, 0], X[:, 1], s=6, c=y, cmap='coolwarm', vmin=0, vmax=1)
# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Class')
plt.show()



pd.DataFrame(y_test).describe()
pd.DataFrame(y_test).value_counts()

###

# train the model on train set
model = SVC(kernel='linear')
model.fit(X_train, y_train)
  
# print prediction results
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,1]+ cm[1,0]+cm[0,0])
print('accuracy' , accuracy)


## busqueda de parametros para el caso lineal ###
from sklearn.model_selection import GridSearchCV
 
# definimos los rangos de los parametros 
param_grid_lineal = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 100, 1000] } 
grid = GridSearchCV(SVC(kernel='linear'), param_grid_lineal, refit = True, cv=10, verbose = 3)
  
# ENTRENAMOS EN TRAIN Y BUSCAMOS EN TRAIN
resultados = grid.fit(X_train, y_train)

#############################

grid_predictions = grid.predict(X_test)
  
# print classification report
print(classification_report(y_test, grid_predictions))
cm = confusion_matrix(y_test, grid_predictions)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,1]+ cm[1,0]+cm[0,0])
print('accuracy' , accuracy)



from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

sm = SMOTE(random_state=42)
#sm = SMOTEENN(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


### DIBUJAMOS
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_res[:, 0], X_res[:, 1], s=6, c=y_res, cmap='coolwarm', vmin=0, vmax=1)
# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Class')
plt.show()



pd.DataFrame(y_res).describe()
pd.DataFrame(y_res).value_counts()

#[X_train, X_test, y_train, y_test] = train_test_split(X_res, y_res, test_size = 0.30, random_state = 101)

model = SVC(kernel='linear')
model.fit(X_res, y_res)
  
# print prediction results
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,1]+ cm[1,0]+cm[0,0])
print('accuracy' , accuracy)

