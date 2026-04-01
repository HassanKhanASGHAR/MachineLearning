# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:09:48 2023

@author: dagom
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import KFold, ShuffleSplit, LeaveOneOut, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import Binarizer
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot

""" Semilla """
seed = 99

## DATOS ##


#df = pd.read_csv("c:/Users/dagom/Documents/docencia/DOCENCIA_FINAL_2023_PHYTON_ML_DANI/2_Introduccion_y_SVM/SAheart.csv", sep=",", decimal=".")
data = pd.read_csv("C:/Users/user.DESKTOP-EHHFBKM/Desktop/master_2024/ONLINE_SVM_ESEMBLE_FINAL_JUNIO_2024/1_SVM/Programaas_y_datos/SAheartbis.csv", sep=",", decimal=".")


#data = pd.read_csv("D:/documentos_en_D/docencia_master/SVM_emsemble_2024_dani/2_Introduccion_y_SVM (1)/2_Introduccion_y_SVM/datos_practica_miss.csv", sep=",", decimal=".")
#data.dropna(inplace=True)
#y=pd.DataFrame(data["NObeyesdad"])
#X=data.drop(columns="NObeyesdad")


data = data.drop(columns=["Unnamed: 0"]) ## depende de la versión la exporta

y=pd.DataFrame(data["chd"])
X=data.drop(columns="chd")


[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.30, random_state = 101)


## 
# Crear un pipeline con normalización y el clasificador RandomForest
#pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=seed))
# Lista de columnas categóricas
#categorical_features = ['categorical_column1', 'categorical_column2']

# Definir la transformación de columnas para aplicar OneHotEncoder a columnas categóricas
#preprocessor = ColumnTransformer(
#    transformers=[
#        ('num', StandardScaler(), ['numeric_column1', 'numeric_column2']),
#        ('cat', OneHotEncoder(), categorical_features)
#    ])

# Crear el pipeline con preprocesamiento y RandomForestClassifier
#pipeline_with_onehot = make_pipeline(preprocessor, RandomForestClassifier(random_state=seed))

### MODELOS ##

models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier(random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('RFC', RandomForestClassifier(random_state=seed)))
models.append(('SVM', SVC(probability=True)))

###############

results = []
names = []
prediciones = []


#name='SVM'
#model=SVC(probability=True)
for name, model in models:
 
    pipeline = make_pipeline(model) ## en realidad no hacemos nada de pipeline pero lo dejo preparado
    pipeline.fit(X_train,y_train)
    prediction=pipeline.predict_proba(X_test)[:,1] ## me quedo con la probabilidad del si
    prediction=pd.DataFrame(prediction, columns=[name])
    
    prediciones.append(prediction)
    accuracy=pipeline.score(X_test, y_test)
   # kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
   #cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')
    results.append(accuracy)
    names.append(name)
print('Resultados', results)    

## suele ser interesante mirar las correlaciones entre las predicciones. A priori cuanto menos correlados 
# son las predicciones mejor es el ensamblado

## visualiza
# Calcular las correlaciones entre las predicciones
# Pimero creamos un dataframe con las predicciones

df=pd.concat(prediciones, axis=1) # con esta sentencia transformamos la lista de dataframes en un unico dataframe
correlation_matrix = pd.DataFrame(df.corr())

# Crear una matriz de correlación con colores
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
##


### ensemble basico 
## agregamos las probabilidades de los diferentes algoritmos para ver potencialmente 
ensemble1= pd.DataFrame((prediciones[0]["LR"] +  prediciones[1]["LDA"])/2)
ensemble1=ensemble1.set_axis(['ensemble1'], axis=1)
ensemble2= pd.DataFrame((prediciones[0]["LR"] +  prediciones[2]["KNN"])/2)
ensemble2=ensemble2.set_axis(['ensemble2'], axis=1)
ensemble3= pd.DataFrame((prediciones[0]["LR"] +  prediciones[3]["DTC"])/2)
ensemble3=ensemble3.set_axis(['ensemble3'], axis=1)
ensemble4= pd.DataFrame((prediciones[0]["LR"] +  prediciones[4]["NB"])/2)
ensemble4=ensemble4.set_axis(['ensemble4'], axis=1)
ensemble5= pd.DataFrame((prediciones[0]["LR"] +  prediciones[5]["RFC"])/2)
ensemble5=ensemble5.set_axis(['ensemble5'], axis=1)
ensemble6= pd.DataFrame((prediciones[0]["LR"] +  prediciones[6]["SVM"])/2)
ensemble6=ensemble6.set_axis(['ensemble6'], axis=1)
ensemble7= pd.DataFrame((prediciones[0]["LR"] +  prediciones[1]["LDA"] + prediciones[6]["SVM"]))  /3
ensemble7=ensemble7.set_axis(['ensemble7'], axis=1)

# Calcula el ensamble tomando el máximo valor entre los tres clasificadores

ensemble8 = pd.DataFrame({'Prediccion_Max_Ensamble': np.maximum.reduce([df.values.flatten() for df in prediciones])})

# Calcula el ensamble tomando el mínimo valor entre los tres clasificadores
ensemble9 = pd.DataFrame({'Prediccion_Min_Ensamble': np.minimum.reduce([df.values.flatten() for df in prediciones])})



Resultados_finales=pd.concat((ensemble1, ensemble2, ensemble3,ensemble4, ensemble5, ensemble6, ensemble7, ensemble8, ensemble9), axis=1 )

from sklearn.preprocessing import Binarizer

transformer = Binarizer(threshold=0.5).fit(Resultados_finales)  # fit does nothing.
predicciones_ensemble=pd.DataFrame(transformer.transform(Resultados_finales))



## a ojo ojo=pd.concat([predicciones_ensemble.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
# Calculamos las curvas ROC y accuracy de los ensembles

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

y_test.replace(('Si', 'No'), (1, 0), inplace=True)

accu = []
for k in range(9):
  cm = confusion_matrix(y_test.iloc[:,0], predicciones_ensemble.iloc[:,k])
  accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,1]+ cm[1,0]+cm[0,0]) 
  print('La precision para el ensemble' ,k+1, 'es', accuracy )
  print('El area bajo la curva para ensemble 1 es' , roc_auc_score(y_test, predicciones_ensemble.iloc[:,k]))
  accu.append(accuracy)

## Mejora la mezcla a los originales ?
print('El area bajo la curva para LR es' , roc_auc_score(y_test, prediciones[0]["LR"]))
print('El area bajo la curva para LDA es' , roc_auc_score(y_test, prediciones[1]["LDA"]))
print('El area bajo la curva para KNN es' , roc_auc_score(y_test, prediciones[2]["KNN"]))
print('El area bajo la curva para DTC es' , roc_auc_score(y_test, prediciones[3]["DTC"]))
print('El area bajo la curva para NB es' , roc_auc_score(y_test, prediciones[4]["NB"]))
print('El area bajo la curva para RFC es' , roc_auc_score(y_test, prediciones[5]["RFC"]))
print('El area bajo la curva para SVM es' , roc_auc_score(y_test, prediciones[6]["SVM"]))

# Pintamos las curvas ROC


ensemble1_fpr, ensemble1_tpr, _ = roc_curve(y_test, ensemble1)
ensemble2_fpr, ensemble2_tpr, _ = roc_curve(y_test, ensemble2)
ensemble3_fpr, ensemble3_tpr, _ = roc_curve(y_test, ensemble3)
ensemble4_fpr, ensemble4_tpr, _ = roc_curve(y_test, ensemble4)
ensemble5_fpr, ensemble5_tpr, _ = roc_curve(y_test, ensemble5)
ensemble6_fpr, ensemble6_tpr, _ = roc_curve(y_test, ensemble6)
ensemble7_fpr, ensemble7_tpr, _ = roc_curve(y_test, ensemble7)
ensemble8_fpr, ensemble8_tpr, _ = roc_curve(y_test, ensemble8)
ensemble9_fpr, ensemble9_tpr, _ = roc_curve(y_test, ensemble9)

fig, ax = plt.subplots(figsize=(10, 10)) 

pyplot.plot(ensemble1_fpr, ensemble1_tpr, linestyle='--',  label='Ensemble 1')
pyplot.plot(ensemble2_fpr, ensemble2_tpr, linestyle='--', label='Ensemble 2')
pyplot.plot(ensemble3_fpr, ensemble3_tpr, linestyle='--', label='Ensemble 3')
pyplot.plot(ensemble4_fpr, ensemble4_tpr, linestyle='--', label='Ensemble 4')
pyplot.plot(ensemble5_fpr, ensemble5_tpr, linestyle='--', label='Ensemble 5')
pyplot.plot(ensemble6_fpr, ensemble6_tpr, linestyle='--', label='Ensemble 6')
pyplot.plot(ensemble7_fpr, ensemble7_tpr, linestyle='--', label='Ensemble 7')
pyplot.plot(ensemble7_fpr, ensemble7_tpr, linestyle='--', label='Ensemble 8')
pyplot.plot(ensemble7_fpr, ensemble7_tpr, linestyle='--', label='Ensemble 9')

# Etiquetas de los ejes
pyplot.xlabel('Tasa de Falsos Positivos')
pyplot.ylabel('Tasa de Verdaderos Positivos')
pyplot.legend()
pyplot.show()

