# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:31:09 2023

@author: dagom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sns.set_style('darkgrid')
np.set_printoptions(precision=2) 
warnings.filterwarnings("ignore")

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
""" Semilla """
seed = 99


#data = pd.read_excel("c:/Users/dagom/Documents/docencia/DOCENCIA_FINAL_2023_PHYTON_ML_DANI/2_Introduccion_y_SVM/Chicago_Crimes_2012_to_2017.csv")
data = pd.read_csv("C:/Users/user.DESKTOP-EHHFBKM/Desktop/master_2024/2_Introduccion_y_SVM/Chicago_Crimes_2012_to_2017.csv")


data = pd.DataFrame(data)
variables = data.columns.values 

data = data.drop(columns=[variables[0]])  ## quitamos la primera variable que no dice nada

print(f'Número de filas: {data.shape[0]}, Número de columnas:{data.shape[1]}') 
data.head()
data.info()

#################################
#### FE y depuración de datos ##
#################################

## Primero vemos cuantos missing tenemos en cada variable ##
data.isnull().sum()

data = data[~data["Latitude"].isnull()] ## eliminamos los datos mising 
data.isnull().sum()

data = data[~data["Location Description"].isnull()] ## eliminamos los datos mising 
data.isnull().sum()  ## sin missing ya (podriamos haberlos imputado pero en este caso no tiene sentido)

## si quiero quitarme todos de un solo paso
#data = data.dropna(axis=1)

###operaciones con fechas interesantes ## es clave para este problema

from datetime import datetime
import calendar


for n in data.index:
 aux= data["Date"][n]  
 data["Date"][n]= datetime.strptime(aux, '%m/%d/%Y %I:%M:%S %p')

data.info()
data.head()
#datetime.strptime(date_string, format)

## de la variable tiempo sacamos 5 variables mas
# Primero me genero 5 variables nuevas

data["hour"]= data["Latitude"]
data["minute"]= data["Latitude"]
data["mes"]= data["Latitude"]
data["dia"]= data["Latitude"]
data["dayweek"]= data["Latitude"]


for n in data.index:
 aux= data["Date"][n]  
 data["hour"][n] = aux.hour 
 data["minute"][n] = aux.minute 
 data["mes"][n] = aux.month
 data["dia"][n] = aux.day
 data["dayweek"][n] = calendar.day_name[aux.weekday()]


### ahora quitamos fecha 

data = data.drop(columns=["Date"])

## separamos la variable target de las variables eplicativas si quisieramos hacer un modelo predictivo
variables = data.columns.values 
X = data[[variables[2], variables[3], variables[4], variables[5], variables[6],variables[7],variables[8],variables[9],variables[10] ]]
y = data[variables[1]]


####

X.info()
X.describe()
y.value_counts()
############ ANALISIS DESCRIPTIVO CONTRA TARGET (IGUAL HAY QUE SEGUIR TRANSFORMANDO)
tabla_target=pd.DataFrame(data[variables[1]].value_counts())
table_location=pd.DataFrame(data["Location Description"].value_counts())
table_dayweek=pd.DataFrame(data["dayweek"].value_counts())
table_dia=pd.DataFrame(data["dia"].value_counts())
table_hour=pd.DataFrame(data["hour"].value_counts())
table_beat=pd.DataFrame(data["Beat"].value_counts())

#############
data.info()
## bidimensional entre todas contra target algunas cosas se pueden observar ya.....
sns.set_theme(style="ticks")
sns.pairplot(data, hue="Arrest")
## se ven cosas pero vamos a diferencias categoricas, continuas y binarias en tres grupos para ver mejor....

col_cat = data[['Location Description', 'Beat', 'District', 'dayweek']]
col_num = data[['Latitude', 'Longitude', 'hour', 'minute', 'mes', 'dia']]
col_bin = data[['Arrest']]


###

#### visualizacion unidimensional categoricas
count=1
plt.subplots(figsize=(10, 8))
for i in col_cat.columns:
    plt.subplot(2,2,count)
    sns.countplot(x=col_cat[i], data = data, dodge = False)
    count+=1

plt.show()

## de los resultados que acabamos de ver tenemos dos variables categoricas 
# que deben ser agrupadas o modificadas... porque tienen muchisimas categorias
## beat debe tener un tratamiento continuo ser eliminado o agrupado por un experto ...


count=1
plt.subplots(figsize=(40, 15))
for i in col_num.columns:
    plt.subplot(2,3,count)
    sns.kdeplot(x=col_num[i], data=data,  shade = True)
    count+=1

plt.show()   

#### visualizacion bidimensional de las continuas frente al target 
count=1
plt.subplots(figsize=(40, 15))
for i in col_num.columns:
    plt.subplot(2,3,count)
    sns.kdeplot(x=col_num[i], data=data, hue=data['Arrest'], shade = True)
    count+=1

plt.show()   

## box plot 
count=1
plt.subplots(figsize=(12, 5))
for i in col_num.columns:
    plt.subplot(2,3,count)
    sns.boxplot(x=col_num[i], data=data, hue=data['Arrest'])
    count+=1

plt.show()  

## no se hace analisis de outliers ## 
############## visualizacion realizada ####
### Finalmente se generan dummies para las variables no continuas
## se puede estudiar reducir el numero de categorias para la variable Localizacion
print(table_location)
data= data.rename({'Location Description': 'Location'}, axis=1)
data['Location'] = data.Location.apply(lambda x: x if x in ['STREET','RESIDENCE','APARTMENT','SIDEWALK', 'PARKING LOT/GARAGE(NON.RESID.)'] else 'Otros')

table_location=pd.DataFrame(data["Location"].value_counts())
print(table_location)


# Creamos dummies solo para la red, en el resto sea realizará en el pipeline
data_dummies = pd.get_dummies(data,columns=['Location', 'dayweek'], drop_first= True)
data_dummies.head()

###################
## separamos la variable target de las variables eplicativas 
variables = data_dummies.columns.values 
X = data_dummies.drop('Arrest', axis=1)
y = data_dummies["Arrest"]

#########################################################
################## UNA SIMPLE REGRESIÓN LOGISTICA !!!!!!
#########################################################
# PASO 1. Dividimos train y test al 30 & 70

[X_train, X_test,y_train,y_test] = train_test_split(X,y,test_size=0.3)


#### PASO 2 SELECCION DE VARIABLES (ME LO VOY A SALTAR POR AHORA)

model = ExtraTreesClassifier()
model.fit(X_train, y_train)
model.feature_importances_

Importancia_tree=pd.DataFrame(zip(model.feature_importances_, X_train.columns), columns=['importance', 'feature'])\
    .sort_values('importance', ascending=False).head()

    
## PASO 3 DECIDIMOS EL MODELO (AQUI SIN TUNEAR NI BUSQUEDA PARAMETRICA NI NADA...)

lr = LogisticRegression()
lr.fit(X_train,y_train)

lr.classes_    
lr.intercept_
lr.coef_
lr.predict_proba(X_test)
lr.predict(X_test)
lr.score(X_test, y_test)
lr.score(X_train, y_train)
confusion_matrix(y_test, lr.predict(X_test))

confusion_matrix(y_train, lr.predict(X_train))

cm = confusion_matrix(y_test, lr.predict(X_test))

print(classification_report(y_test, model.predict(X_test)))

## Paso 4 con validacion cruzada 
from sklearn.model_selection import cross_val_score
model=lr
results = cross_val_score(estimator=model, X=X, y=y, cv=5)
print(results)



import folium
from folium.plugins import HeatMap

# Supongamos que tienes un DataFrame llamado 'df' con columnas 'latitud' y 'longitud'
# Asegúrate de tener instaladas las bibliotecas necesarias con: pip install folium pandas


# Crear un mapa centrado en las coordenadas iniciales
df=data
df.columns

mapa = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

# Convertir las coordenadas a una lista de listas
coordenadas = df[['Latitude', 'Longitude']].values.tolist()

# Agregar un mapa de calor
HeatMap(coordenadas).add_to(mapa)

# Guardar el mapa como un archivo HTML o mostrarlo en el cuaderno (dependiendo de tu entorno)
mapa.save('mapa_calor.html')  # Guarda el mapa como un archivo HTML
mapa


#########################
import folium
from folium.plugins import MarkerCluster, HeatMap

# Crear un mapa centrado en las coordenadas iniciales
df = data  # Asumo que ya tienes el DataFrame 'data' con las columnas 'Latitude', 'Longitude' y 'arrest'
mapa = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

# Convertir las coordenadas y la variable 'arrest' a una lista de listas
datos_mapa = df[['Latitude', 'Longitude', 'Arrest']].values.tolist()

# Inicializar dos listas para almacenar las coordenadas de arrestos y no arrestos
coordenadas_arresto = []
coordenadas_no_arresto = []

for dato in datos_mapa:
 
    if dato[2] == True:
       
        coordenadas_arresto.append([dato[0], dato[1]])
    else:
        coordenadas_no_arresto.append([dato[0], dato[1]])

# Create a map for arrests
mapa_arresto = folium.Map(location=[0, 0], zoom_start=2)
arrest_cluster = MarkerCluster(name='Arresto', overlay=True, control=True, icon_create_function=None)

for coord in coordenadas_arresto:
    folium.Marker(location=coord, popup='Arresto', icon=folium.Icon(color='red')).add_to(arrest_cluster)

arrest_cluster.add_to(mapa_arresto)
HeatMap(coordenadas_arresto).add_to(mapa_arresto)

# Save the map for arrests
mapa_arresto.save('mapa_arresto.html')

# Create a map for no arrests
mapa_no_arresto = folium.Map(location=[0, 0], zoom_start=2)
no_arrest_cluster = MarkerCluster(name='No Arresto', overlay=True, control=True, icon_create_function=None)

for coord in coordenadas_no_arresto:
    folium.Marker(location=coord, popup='No Arresto', icon=folium.Icon(color='green')).add_to(no_arrest_cluster)

no_arrest_cluster.add_to(mapa_no_arresto)
HeatMap(coordenadas_no_arresto).add_to(mapa_no_arresto)

# Save the map for no arrests
mapa_no_arresto.save('mapa_no_arresto.html')