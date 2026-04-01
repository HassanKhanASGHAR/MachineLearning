# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:03:54 2023

@author: dagom
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# importing machine learning models for prediction
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
# importing voting classifier
from sklearn.ensemble import VotingClassifier
seed = 99

## DATOS ##

#df = pd.read_csv("c:/Users/dagom/Documents/docencia/DOCENCIA_FINAL_2023_PHYTON_ML_DANI/2_Introduccion_y_SVM/SAheart.csv", sep=",", decimal=".")
data = pd.read_csv("D:/documentos_en_D/docencia_master/SVM_emsemble_2024_dani/2_Introduccion_y_SVM (1)/2_Introduccion_y_SVM/SAheartbis.csv", sep=",", decimal=".")


y=pd.DataFrame(data["chd"])
X=data.drop(columns="chd")


[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.30, random_state = 101)



# initializing all the model objects with default parameters
model_1 = LogisticRegression()
model_2 = XGBClassifier()
model_3 = RandomForestClassifier()

# Making the final model using voting classifier
final_model = VotingClassifier(
	estimators=[('lr', model_1), ('xgb', model_2), ('rf', model_3)], voting='hard')

# training all the model on the train dataset
final_model.fit(X_train, y_train)

# predicting the output on the test dataset
pred_final = final_model.predict(X_test)
 

cm=confusion_matrix(y_test, pred_final)
print(cm)

accuracy_score(y_test, pred_final)

## mejoramos ????
from sklearn.preprocessing import LabelEncoder
# Codifica las etiquetas de clase a valores numéricos o el boost dara error

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train.values.ravel())
y_test_encoded = label_encoder.transform(y_test.values.ravel())

# Inicializa los modelos
model_1 = LogisticRegression()
model_2 = XGBClassifier()
model_3 = RandomForestClassifier()

# Ajusta los modelos a los datos de entrenamiento
model_1.fit(X_train, y_train_encoded)
model_2.fit(X_train, y_train_encoded)
model_3.fit(X_train, y_train_encoded)

# Realiza predicciones en el conjunto de prueba
predicciones_1 = model_1.predict(X_test)
predicciones_2 = model_2.predict(X_test)
predicciones_3 = model_3.predict(X_test)

# Evalúa el rendimiento de los modelos
accuracy_1 = accuracy_score(y_test_encoded, predicciones_1)
accuracy_2 = accuracy_score(y_test_encoded, predicciones_2)
accuracy_3 = accuracy_score(y_test_encoded, predicciones_3)

print("Accuracy Model 1:", accuracy_1)
print("Accuracy Model 2:", accuracy_2)
print("Accuracy Model 3:", accuracy_3)

# Puedes utilizar otras métricas de evaluación según tus necesidades
# Por ejemplo, classification_report para obtener precision, recall, f1-score, etc.
report_1 = classification_report(y_test_encoded, predicciones_1)
report_2 = classification_report(y_test_encoded, predicciones_2)
report_3 = classification_report(y_test_encoded, predicciones_3)

print("Classification Report Model 1:\n", report_1)
print("Classification Report Model 2:\n", report_2)
print("Classification Report Model 3:\n", report_3)

