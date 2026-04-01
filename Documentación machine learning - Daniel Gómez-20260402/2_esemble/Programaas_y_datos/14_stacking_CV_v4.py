# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:11:04 2023

@author: dagom
"""

# importing utility modules
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import StackingClassifier
import pandas as pd
# importing machine learning models for prediction
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression,LogisticRegression
### si lo queremos en validacion cruzada 
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix 
# importing train test split


seed = 99

## DATOS ##
data = pd.read_csv("D:/documentos_en_D/docencia_master/SVM_emsemble_2024_dani/2_Introduccion_y_SVM (1)/2_Introduccion_y_SVM/SAheartbis.csv", sep=",", decimal=".")
y=pd.DataFrame(data["chd"])
y.replace(('Si', 'No'), (1, 0), inplace=True)
X=data.drop(columns="chd")
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.30, random_state = 101)


# Inicializar los modelos base
model_1 = RandomForestClassifier(random_state=42)
model_2 = LogisticRegression(random_state=42)
model_3 = KNeighborsClassifier()

# putting all base model objects in one list
all_models = [('rf', model_1), ('lr', model_2), ('knn', model_3)]


# Inicializar el modelo de stacking
stacking_model = StackingClassifier(
    estimators=[('rf', model_1), ('lr', model_2), ('knn', model_3)],
    final_estimator=RandomForestClassifier(random_state=42)
)


# Crear un objeto StratifiedKFold para la validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Obtener predicciones de la validación cruzada
y_pred_cv_train = cross_val_predict(stacking_model, X_train, y_train, cv=cv, method='predict')
y_pred_cv_test = cross_val_predict(stacking_model, X_test, y_test, cv=cv, method='predict')


# Calcular métricas de calidad en train
accuracy = accuracy_score(y_train, y_pred_cv_train)
precision = precision_score(y_train, y_pred_cv_train)
recall = recall_score(y_train, y_pred_cv_train)
f1 = f1_score(y_train, y_pred_cv_train)

# Imprimir las métricas
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


# Calcular métricas de calidad en train
accuracy = accuracy_score(y_test, y_pred_cv_test)
precision = precision_score(y_test, y_pred_cv_test)
recall = recall_score(y_test, y_pred_cv_test)
f1 = f1_score(y_test, y_pred_cv_test)

# Imprimir las métricas
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')



################################
### ha merecido la pena ???
#################################


# Obtener los clasificadores internos
base_classifiers = stacking_model.estimators_

# Obtener las características metaaprendidas por cada clasificador base
X_train_meta = stacking_model.transform(X_train)

# Evaluar el rendimiento de cada clasificador base individualmente
for (name, clf) in stacking_model.named_estimators_.items():
    y_pred_base = clf.predict(X_test)
    print(f"\nResultados del Clasificador Base {name}:")
    print(classification_report(y_test, y_pred_base))



