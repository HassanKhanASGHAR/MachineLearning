# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:31:09 2023

@author: dagom
"""

## empezamos con el ejemplo del titanic modificado 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sns.set_style('darkgrid')
np.set_printoptions(precision=2) 
warnings.filterwarnings("ignore")

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
""" Semilla """
seed = 99

## ALGUNAS UTILIDADES SOBRE EL MANEJO DE DATOS SEGURAMENTE SEA REDUNDANTE CON LO VISTO ANTERIORMENTE POR VOSOTROS


data = pd.read_excel("C:/Users/user.DESKTOP-EHHFBKM/Desktop/master_2024/2_Introduccion_y_SVM/titanic.xlsx")
data = pd.read_excel( "C:/Users/user.DESKTOP-EHHFBKM/Desktop/master_2024/2_Introduccion_y_SVM/titanic.xlsm")

data = data.drop(data.columns[0], axis=1) # eliminamos la primera columna que no tiene sentido

 ## data.shape me da un vector (tuple) de dos dimensiones con las filas y las columnas del dataframe

print(f'Número de filas: {data.shape[0]}, Número de columnas:{data.shape[1]}') 
data.head()
data.dtypes
####
# Crear un histograma utilizando Seaborn
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Fare', hue='Embarked', bins=30, kde=True)

# Añadir etiquetas y título
plt.xlabel('Tarifa (Fare)')
plt.ylabel('Frecuencia')
plt.title('Histograma de Tarifas según el Puerto de Embarque')

# Mostrar el gráfico
plt.show()

# Crear el boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Fare', data=data)
plt.title('Boxplot de Fare según la supervivencia')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.show()


# Crear el boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Survived', y='Fare', hue='Sex', data=data)
plt.title('Boxplot de Fare según la supervivencia y el género')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.legend(title='Sex', loc='upper right')
plt.show()

###########################################################################
## Paso 1 ## Missing
## Paso 1. vemos cuantos missing tenemos en cada variable ##
data.isnull().sum() 

import seaborn as sns
# Identificamos los missing values visualmente
sns.heatmap(data.isnull(), cbar=False)
## vemos que edad, cabin y Embarked tienen valores perdidos. Cabin tiene muchos valores perdidos
## df.fillna(df.mean(), inplace=True)

###########################################################################
## Paso 2 Codificacion, Imputación, o eliminacion de datos perdidos
## en ocasiones la información relevante es si se ha perdido un dato o no
## en otros casos merece la pena imputar su valor mientras que otras veces merece la pena eliminarlo

###############################################
## 2.1 Modificacion de la variable Cabin. Porque Hacemos esto?

#Se llenan los valores nulos (NaN) de la columna 'Cabin' de data con el valor 0.
data['Cabin'] = data['Cabin'].fillna(0)
#Nueva columna hasCabin, que toma valores binarios 0 y 1, en función de si el pasajero tiene o no un número de cabina.
# La función lambda define que si el valor de "Cabin" es 0 el valor de la columna hasCabin toma valor 0, 
# en caso contrario toma  valor 1.
data['hasCabin'] = data['Cabin'].apply(lambda x: 0 if x==0 else 1)

# tabla_frecuencias = pd.crosstab(data['hasCabin'], data['Survived'])

#Se eliminan de data las columnas 'PassengerId', 'Name', 'Cabin', 'Ticket'
data = data.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'])


key_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'hasCabin']

data.head()
### Paso xx. Esto es solo para ver algunas modificaciones que podriamos hacer a la base de datos.
## algunas modificaciones que pueden ser interesantes
#Se pide en esta celda, modificar las variables Title, Parch y SibSp, donde Title tome solo los valores Mr, Mrs, Miss y Otros; 
# Parch y SibSp toman solo los valores 0, 1 o 2 (donde 2 incluye 2 o más).
#

data['SibSp'] = data.SibSp.apply(lambda x: 2 if x>=2 else x)
data['Parch'] = data.Parch.apply(lambda x: 2 if x>=2 else x)
# La función lambda define que si el valor de "Title" es uno de estos tres ['Mr','Mrs','Miss'],
# se mantendrá el valor original. De lo contrario, se reemplazará con el valor "Otros".
data['Title'] = data.Title.apply(lambda x: x if x in ['Mr','Mrs','Miss'] else 'Otros')
#data['Title2'] = data.Title.apply(lambda x: 'Casados' if x in ['Mr','Mrs'] else 'Otros')

# Se eliminan las 2 filas que tienen valores nulos (NaN) en esta columna.
data = data[~data.Embarked.isnull()]
# Se eliminan todos los duplicados.
data = data.drop_duplicates(inplace=False)

############################################3
###### 2.2 imputacion de datos perdidos
imputer = KNNImputer(n_neighbors=3, metric='nan_euclidean')
# Imputamos los valores perdidos en el DataFrame 
## Para imputar con Knn necesitamos tranformar las variables en numeros
le = LabelEncoder()
# Codificamos las variables categóricas como numéricas
data_encoder = data.copy()

for col in data_encoder.columns:
    if data_encoder[col].dtype == 'object':  # Si la columna contiene valores categóricos
        le = LabelEncoder()         # Creamos una instancia de LabelEncoder
        data_encoder[col] = le.fit_transform(data_encoder[col])  # Codificamos la columna como numérica

## ahora podemos inputar por ejemplo con el KNN

data_imputed = pd.DataFrame(imputer.fit_transform(data_encoder))
## recuperamos los nombres y tenemos  nuestra base de datos depurada  
data_imputed=data_imputed.set_axis([key_cols], axis=1)
## datos sin missing y encoder


################################
######### CREAR PIPELINE ### 
## PROCEDIMIENTO MUUY COMODO PARA HACER TODO EN UN UNICO PASO.
## UN PIPELINE COGE UNA BASE DE DATOS Y HACE DEPURACION, EDA, ETC TODO EN UNO.

#

#ColumnTransformer: aplica diferentes transformaciones a diferentes columnas de data y 
#generar un nuevo DataFrame "Transformed data".
transformer1 = [
    ('KNNImputer', KNNImputer(n_neighbors=5,  weights='uniform', metric='nan_euclidean'), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
    ('PowerTransformerYJ', PowerTransformer(method='yeo-johnson'), ['Fare']),
    ('encoder1', OneHotEncoder(drop='if_binary'), ['Sex']),
    ('encoder2', OneHotEncoder(drop='first'), ['Embarked']),
    ('encoder3', OneHotEncoder(), ['Title']),
    ('encoder4', OneHotEncoder(), ['Parch'])
]

transformer2= [
    ('KNNImputer', KNNImputer(n_neighbors=5,  weights='uniform', metric='nan_euclidean'), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
    
]

col_transformer = ColumnTransformer(transformer1, remainder='drop')

#KNNImputer :  técnica de imputación de valores faltantes que utiliza el algoritmo de los k vecinos más cercanos 
#para estimar los valores faltantes basándose en los valores de las columnas cercanas. 

#PowerTransformer: transforma las variables numéricas 
# con el objetivo de mejorar la distribución y normalidad de los datos.

#OneHotEncoder: transformar una variable categórica  en una representación numérica binaria.

#remainder=drop, por lo que las columnas no transformadas se eliminan del conjunto de datos transformado.

ctransformed = col_transformer.fit_transform(data)

##############pipeline para modelos de ML

models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier(random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('RFC', RandomForestClassifier(random_state=seed)))
models.append(('SVM', SVC()))

X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'hasCabin']]
y = data['Survived']

def boxplots_algorithms(results, names):
    
    plt.figure(figsize=(8,8))
    plt.boxplot(results)
    plt.xticks(range(1,len(names)+1), names)
    plt.show()

###############

results = []
names = []

for name, model in models:
    scaler = RobustScaler()
    pipeline = make_pipeline(col_transformer, scaler, model)
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

boxplots_algorithms(results, names)

# • Pipeline : Encadena múltiples pasos (ColumnTransformer, RobustScaler y el Modelo); 
# la salida de cada paso se usa como entrada para el siguiente paso.
# • Validación cruzada : Kfold se utiliza para dividir los datos  en 10 carpetas y en cada iteración utiliza uno 
# de estos carpetas como conjunto de prueba y los demás como conjunto de entrenamiento. El argumento shuffle=True 
# indica que los datos se mezclarán antes de dividirlos en pliegues y random_state=seed para que la división sea
# determinista y reproducible. cross_val_score para realizar la validación cruzada y obtener el score de cada modelo. 

# SVM es el mejor modelo, con un score de 0.7980


