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

from sklearn.metrics import classification_report, confusion_matrix 
# importing train test split

# Import Support Vector Classifier
from sklearn.svm import SVC

seed = 99

## DATOS ##
data=pd.read_csv('D:/documentos_en_D/docencia_master/SVM_emsemble_2024_dani/2_Introduccion_y_SVM (1)/2_Introduccion_y_SVM/SAheartbis.csv')

#data = pd.read_csv("D:/documentos_en_D/docencia_master/SVM_emsemble_2024_dani/2_Introduccion_y_SVM (1)/2_Introduccion_y_SVM/SAheartbis.csv", sep=",", decimal=".")
y=pd.DataFrame(data["chd"])
y.replace(('Si', 'No'), (1, 0), inplace=True)
X=data.drop(columns="chd")
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2, random_state=42)

#######
# Convierte y_train a un array de NumPy
#from sklearn.preprocessing import LabelEncoder
#label_encoder = LabelEncoder()
#y_train =label_encoder.fit_transform(y_train).ravel()
#y_test=label_encoder.fit_transform(y_test).ravel()

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data
y = iris.target


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


# Inicializar el modelo de stacking
stacking_model = StackingClassifier(
    estimators=[('rf', model_1), ('lr', model_2), ('knn', model_3)],
    final_estimator=SVC(probability=True)
)


### vamos con el stacking !!!!!
# Inicializar el modelo final de clasificación
final_model = RandomForestClassifier(random_state=42)

# Entrenar el modelo de stacking
stacking_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = stacking_model.predict(X_test)

# Mostrar el classification report
print(classification_report(y_test, y_pred))

### si lo queremos en validacion cruzada 
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Crear un objeto StratifiedKFold para la validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Obtener predicciones de la validación cruzada
y_pred_cv_train = cross_val_predict(stacking_model, X_train, y_train, cv=cv, method='predict')
y_pred_cv_test = cross_val_predict(stacking_model, X_test, y_test, cv=cv, method='predict')

# Agregar las listas como columnas en lugar de crear un DataFrame directamente
df_cv_results = pd.DataFrame()
df_cv_results['True'] = y_train
df_cv_results['Predicted'] = y_pred_cv_train

# Calcular métricas de calidad en train
accuracy = accuracy_score(y_train, y_pred_cv_train)
precision = precision_score(y_train, y_pred_cv_train) # no vamos a poder usarlo porque tenemos mas de dos clases
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





#################### CON MAS OPCIONES #############

# Configurar el StackingClassifier
stacking_model = StackingClassifier(
    estimators=[('rf', model_1), ('lr', model_2), ('knn', model_3)],
    final_estimator=RandomForestClassifier(random_state=42),
    stack_method='auto',  # Puedes ajustar esto según tus necesidades
    cv=5,  # Validación cruzada interna
    passthrough=True  # Pasa las características originales al clasificador final
)

# Entrenar el modelo de stacking
stacking_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = stacking_model.predict(X_test)

# Mostrar el classification report
print(classification_report(y_test, y_pred))


######################### ahora vamos a tunear un poco el metaclasificador. 
## Es igual a como se hace con cualquier modelo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score


# Inicializar los clasificadores base
clf1 = RandomForestClassifier(random_state=42)
clf2 = LogisticRegression(random_state=42, max_iter=1000)
clf3 = KNeighborsClassifier()

# Inicializar el clasificador final
final_estimator = LogisticRegression(random_state=42, max_iter=1000)

# Configurar el StackingClassifier
stacking_model = StackingClassifier(
    estimators=[('rf', clf1), ('lr', clf2), ('knn', clf3)],
    final_estimator=final_estimator,
    stack_method='auto',
    cv=5,
    passthrough=True
)

# Definir la cuadrícula de parámetros para la búsqueda
param_grid = {'final_estimator__C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Definir la métrica que deseas optimizar (por ejemplo, precisión)
scoring = make_scorer(accuracy_score)

# Realizar la búsqueda de cuadrícula para encontrar la mejor combinación de pesos
grid_search = GridSearchCV(stacking_model, param_grid, scoring=scoring, cv=5)
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo y sus parámetros
best_stacking_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluar el rendimiento del mejor modelo en el conjunto de prueba
y_pred_best = best_stacking_model.predict(X_test)
print("\nResultados del Mejor Modelo:")
print(classification_report(y_test, y_pred_best))
print("Mejores parámetros:", best_params)

##### BLENDING !!!
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize

# Inicializar los modelos base
model_1 = RandomForestClassifier(random_state=42)
model_2 = LogisticRegression(random_state=42)
model_3 = KNeighborsClassifier()
# Poner todos los objetos de modelos base en una lista
all_models = [model_1, model_2, model_3]

# Entrenar los modelos base en el conjunto de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
for model in all_models:
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    print(f"Accuracy for model {model} on Train: {accuracy_train}")
    print(f"Accuracy for model {model} on Test: {accuracy_test}")


# Explorar diferentes combinaciones de pesos manualmente en el conjunto de entrenamiento
best_metric = 0.0
best_weights = None
# Inicializar el diccionario para almacenar los valores de accuracy para cada combinación de pesos
accuracy_values = {}

for w1 in range(0, 101, 10):
    for w2 in range(0, 101 - w1, 10):
        w3 = 100 - w1 - w2
        weights = [w1/100, w2/100, w3/100]

        # Crear y entrenar el modelo de blending final con los pesos actuales
        blending_model = VotingClassifier(
            estimators=[('model_1', model_1), ('model_2', model_2), ('model_3', model_3)],
            voting='soft',
            weights=weights
        )

        blending_model.fit(X_train, y_train)

        # Hacer predicciones en el conjunto de entrenamiento
        y_pred_proba_train = blending_model.predict_proba(X_train)

        # Calcular la métrica en el conjunto de entrenamiento (en este caso, accuracy)
        metric_train = accuracy_score(y_train, y_pred_proba_train.argmax(axis=1))

        # Actualizar la mejor combinación de pesos si se encuentra una métrica mejor en el conjunto de entrenamiento
        if metric_train > best_metric:
            best_metric = metric_train
            best_weights = weights
        # Almacenar la métrica para la combinación de pesos actual
        accuracy_values[tuple(weights)] = metric_train


# Imprimir los valores de accuracy para cada combinación de pesos
print("\nAccuracy Values:")
for weights, accuracy in accuracy_values.items():
    print(f"Weights: {weights}, Accuracy: {accuracy}")

# Imprimir las precisiones para el mejor modelo de blending en ambos conjuntos
print("\nBest Weights:", best_weights)
print("Best Metric on Training Set:", best_metric)

# Crear y entrenar el modelo de blending final con los mejores pesos en todo el conjunto de entrenamiento
final_blending_model = VotingClassifier(
    estimators=[('model_1', model_1), ('model_2', model_2), ('model_3', model_3)],
    voting='soft',
    weights=best_weights
)

final_blending_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred_proba_test = final_blending_model.predict_proba(X_test)

# Calcular la métrica final en el conjunto de prueba (en este caso, accuracy)
final_metric_test = accuracy_score(y_test, y_pred_proba_test.argmax(axis=1))

# Imprimir la métrica final en el conjunto de prueba
print("Final Metric on Test Set:", final_metric_test)