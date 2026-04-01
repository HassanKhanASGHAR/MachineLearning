# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:11:04 2023

@author: dagom
"""
# importing utility modules
import pandas as pd
from sklearn.metrics import mean_squared_error

# importing machine learning models for prediction
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix 
# importing train test split
from sklearn.model_selection import train_test_split
seed = 99

## DATOS ##

## DATOS ##
data = pd.read_csv("D:/documentos_en_D/docencia_master/SVM_emsemble_2024_dani/2_Introduccion_y_SVM (1)/2_Introduccion_y_SVM/SAheartbis.csv", sep=",", decimal=".")


y=pd.DataFrame(data["chd"])
y.replace(('Si', 'No'), (1, 0), inplace=True)
X=data.drop(columns="chd")

############ STACKING MANUAL 

# performing the train test and validation split
train_ratio = 0.70 ### entrenamos los modelos iniciales
validation_ratio = 0.20 ## entrenamos el stacking
test_ratio = 0.10 ## test real para medir la precision del modelo

##  1 - test_ratio es realmente la cantidad que estamos usando para entrenar: modelos iniciales + stacking
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 1 - train_ratio, random_state = 101)

####

# performing test validation split
X_val, X_test, y_val, y_test = train_test_split(
	X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

# initializing all the base model objects with default parameters
model_1 = LinearRegression()
model_2 = xgb.XGBRegressor()
model_3 = RandomForestRegressor()

# training all the model on the train dataset

# training first model
model_1.fit(X_train, y_train)
val_pred_1 = model_1.predict(X_val) ## input train del modelo stacking
test_pred_1 = model_1.predict(X_test) ## output test del modelo stacking

# converting to dataframe
val_pred_1 = pd.DataFrame(val_pred_1)
test_pred_1 = pd.DataFrame(test_pred_1)

# training second model
model_2.fit(X_train, y_train)
val_pred_2 = model_2.predict(X_val)
test_pred_2 = model_2.predict(X_test)

# converting to dataframe
val_pred_2 = pd.DataFrame(val_pred_2)
test_pred_2 = pd.DataFrame(test_pred_2)

# training third model
model_3.fit(X_train, y_train)
val_pred_3 = model_1.predict(X_val)
test_pred_3 = model_1.predict(X_test)

# converting to dataframe
val_pred_3 = pd.DataFrame(val_pred_3)
test_pred_3 = pd.DataFrame(test_pred_3)

# concatenating validation dataset along with all the predicted validation data (meta features)
df_val = pd.concat([val_pred_1, val_pred_2, val_pred_3], axis=1)
df_test = pd.concat([test_pred_1, test_pred_2, test_pred_3], axis=1)

## recuperamos los nombres y tenemos  nuestra base de datos depurada  
df_val=df_val.set_axis(['model1', 'model2', 'model3'], axis=1)
df_test=df_test.set_axis(['model1', 'model2', 'model3'], axis=1)

#### stacking stacking stacking !!!!!!!!
# making the final model using the meta features
final_model = LogisticRegression()
final_model.fit(df_val, y_val)

# getting the final output
final_pred = final_model.predict(df_test)

print(classification_report(y_test, final_pred))

cm = confusion_matrix(y_test, final_pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,1]+ cm[1,0]+cm[0,0])
print('accuracy' , accuracy)


  