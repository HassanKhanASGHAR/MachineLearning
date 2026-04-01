# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 00:34:36 2023

@author: dagom EJEMPLO DEL BANCO !!!!!!
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


## LECTURA DE DATOS

bank=pd.read_csv('c:/Users/dagom/Documents/docencia/DOCENCIA_FINAL_2023_PHYTON_ML_DANI/2_Introduccion_y_SVM/bank-full.csv', sep=';')

variables=bank.columns
print(variables)

X=bank.iloc[:,0:16]
y=pd.DataFrame(bank["y"])

y.describe()
y.value_counts()

## clase del no 39922/45211 ## muy balanceado

## Paso 1. vemos cuantos missing tenemos en cada variable ##
bank.isnull().sum()

# Paso 2. Para poder aplicar SVM las variables deben ser numericas y a ser posibles estandarizadas
from sklearn import preprocessing

col_cat = bank[['job', 'marital','education','default', 'housing','loan','contact','y']]
col_num = bank[['age', 'balance', 'day','duration','pdays','previous']]


scaler = preprocessing.StandardScaler().fit(col_num)
col_num_standarizada = scaler.transform(col_num)

col_num_standarizada.mean(axis=0)
col_num_standarizada.std(axis=0)

col_num_standarizada = pd.DataFrame(col_num_standarizada)


bank_depurada = pd.concat([col_num_standarizada, col_cat],  axis=1) ## aunque asi se pierden los nombres de las variables 

## recuperamos los nombres y tenemos  nuestra base de datos depurada  
bank_depurada=bank_depurada.set_axis(['age', 'balance', 'day' ,'duration','pdays','previous' , 'job', 'marital','education','default', 'housing','loan','contact','y'], axis=1)


## Paso 3. Variables categoricas las pasamos a dummies 

bank_depurada_dummies = pd.get_dummies(bank_depurada,columns=['job', 'marital','education','default', 'housing','loan','contact','y'], drop_first= True)
bank_depurada_dummies.head()


X=bank_depurada_dummies.iloc[:,0:26]
y=pd.DataFrame(bank_depurada_dummies["y_yes"])

y.describe()
y.value_counts()


########### hacemos clasificacion con SVM ####
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.30, random_state = 101)


###

# train the model on train set
model = SVC(kernel='linear')
model.fit(X_train, y_train)
  
# print prediction results
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
print(cm)  ## ojo a esta matriz de confusion !!!!!
accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,1]+ cm[1,0]+cm[0,0])
print('accuracy' , accuracy)
#############################
################## smooooote
#############################




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
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


y_res.describe()
y_res.value_counts()


model = SVC(kernel='linear')
model.fit(X_res, y_res)
  
# print prediction results
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,1]+ cm[1,0]+cm[0,0])
print('accuracy' , accuracy)
