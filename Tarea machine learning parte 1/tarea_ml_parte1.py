#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tarea Machine Learning Parte 1 - Clasificación Binaria
Módulo: Machine Learning
Predicción del color de un coche (Blanco vs No Blanco/Negro)

Estructura:
    1) Análisis y depuración de la base de datos + Feature Engineering
    2) Búsqueda paramétrica del mejor Árbol de Decisión (4 métricas de validación)
    3) Búsqueda paramétrica de Random Forest y XGBoost (según Accuracy)
    4) Comparación completa de modelos
"""

# =============================================================================
# IMPORTS
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score
)
from xgboost import XGBClassifier

sns.set_style('darkgrid')
np.set_printoptions(precision=4)
SEED = 123
np.random.seed(SEED)

# Directory for saving figures
import os
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figuras')
os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# 1) ANÁLISIS Y DEPURACIÓN DE LA BASE DE DATOS + FEATURE ENGINEERING
# =============================================================================
print("=" * 80)
print("1) ANÁLISIS Y DEPURACIÓN DE LA BASE DE DATOS")
print("=" * 80)

# 1.1 Cargar datos
df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'datos_tarea25.xlsx'))
print(f"\nDimensiones originales: {df.shape}")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nTipos de datos:\n{df.dtypes}")

# 1.2 Información general
print(f"\nValores nulos por columna:\n{df.isnull().sum()}")
print(f"\nDuplicados: {df.duplicated().sum()}")

# Eliminar duplicados si existen
n_dup = df.duplicated().sum()
if n_dup > 0:
    df = df.drop_duplicates()
    print(f"Se eliminaron {n_dup} filas duplicadas. Nuevo shape: {df.shape}")

# 1.3 Construcción de la variable objetivo
# Color: White=1 (pintar de blanco), Black=0 (no pintar de blanco)
print(f"\nDistribución de Color:\n{df['Color'].value_counts()}")
print(f"Proporciones:\n{df['Color'].value_counts(normalize=True)}")

df['target'] = (df['Color'] == 'White').astype(int)
df = df.drop('Color', axis=1)
print(f"\nVariable objetivo 'target' creada: 1=White, 0=Black")
print(f"Distribución target:\n{df['target'].value_counts()}")

# 1.4 Limpieza de la variable Levy
# Levy contiene '-' para valores desconocidos
print(f"\nLevy - valores únicos (muestra): {df['Levy'].unique()[:10]}")
print(f"Levy = '-': {(df['Levy'] == '-').sum()} observaciones")
# Convertir '-' a NaN, luego a numérico
df['Levy'] = df['Levy'].replace('-', np.nan)
df['Levy'] = pd.to_numeric(df['Levy'])
print(f"Levy NaN tras conversión: {df['Levy'].isnull().sum()}")

# Imputar Levy con la mediana (robusto a outliers)
levy_median = df['Levy'].median()
df['Levy'] = df['Levy'].fillna(levy_median)
print(f"Levy imputada con mediana: {levy_median}")

# 1.5 Limpieza de Engine volume
# Contiene valores como '2', '2.0 Turbo' -> separar componente numérico y flag turbo
print(f"\nEngine volume - ejemplos: {df['Engine volume'].unique()[:10]}")
df['Turbo'] = df['Engine volume'].str.contains('Turbo', case=False).astype(int)
df['Engine volume'] = df['Engine volume'].str.replace(' Turbo', '', regex=False)
df['Engine volume'] = pd.to_numeric(df['Engine volume'])
print(f"Variable 'Turbo' creada. Distribución:\n{df['Turbo'].value_counts()}")

# 1.6 Limpieza de Mileage
# Contiene ' km' al final
print(f"\nMileage - ejemplos: {df['Mileage'].unique()[:5]}")
df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=False)
df['Mileage'] = pd.to_numeric(df['Mileage'])

# 1.7 Codificación de variables categóricas
# Leather interior: Yes/No -> 1/0
df['Leather interior'] = (df['Leather interior'] == 'Yes').astype(int)

# Wheel: Left wheel / Right-hand drive -> binaria
df['Wheel'] = (df['Wheel'] == 'Left wheel').astype(int)

# Variables categóricas para one-hot encoding (drop_first para evitar multicolinealidad)
cat_cols = ['Manufacturer', 'Category', 'Fuel type', 'Gear box type', 'Drive wheels']
print(f"\nVariables categóricas para one-hot encoding: {cat_cols}")
for col in cat_cols:
    print(f"  {col}: {df[col].unique()}")

df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

# 1.8 Resumen final del dataset
print(f"\nDataset final shape: {df.shape}")
print(f"\nColumnas finales: {df.columns.tolist()}")
print(f"\nEstadísticas descriptivas:\n{df.describe().to_string()}")
print(f"\nTipos finales:\n{df.dtypes}")

# 1.9 Separar X e y
y = df['target']
X = df.drop('target', axis=1)
num_features = X.columns.tolist()
print(f"\nNúmero de features: {X.shape[1]}")
print(f"Features: {num_features}")

# 1.10 Train/Test split (80/20) con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Distribución target en Train:\n{y_train.value_counts(normalize=True)}")
print(f"Distribución target en Test:\n{y_test.value_counts(normalize=True)}")

# 1.11 Análisis exploratorio - gráficos
# Distribución de la variable objetivo
fig, ax = plt.subplots(figsize=(6, 4))
y.value_counts().plot(kind='bar', ax=ax, color=['#2c3e50', '#ecf0f1'], edgecolor='black')
ax.set_title('Distribución de la variable objetivo')
ax.set_xlabel('Color (0=Black, 1=White)')
ax.set_ylabel('Frecuencia')
ax.set_xticklabels(['Black (0)', 'White (1)'], rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig01_distribucion_target.png'), dpi=150)
plt.close()

# Matriz de correlación
fig, ax = plt.subplots(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Matriz de correlación')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig02_correlacion.png'), dpi=150)
plt.close()

# Correlación con la variable target
fig, ax = plt.subplots(figsize=(10, 6))
target_corr = corr['target'].drop('target').sort_values()
target_corr.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Correlación de cada variable con target')
ax.set_xlabel('Correlación')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig03_correlacion_target.png'), dpi=150)
plt.close()

print("\nAnálisis exploratorio completado. Figuras guardadas en 'figuras/'")

# =============================================================================
# 2) BÚSQUEDA PARAMÉTRICA - ÁRBOL DE DECISIÓN
#    Mejor árbol según 4 métodos de validación de bondad de clasificación
# =============================================================================
print("\n" + "=" * 80)
print("2) BÚSQUEDA PARAMÉTRICA - ÁRBOL DE DECISIÓN")
print("=" * 80)

# Definir modelo base y grid de parámetros
dt_base = DecisionTreeClassifier(random_state=SEED)

dt_param_grid = {
    'max_depth': [2, 3, 5, 10, 15, 20],
    'min_samples_split': [5, 10, 20, 50, 100],
    'min_samples_leaf': [3, 10, 30, 50],
    'criterion': ['gini', 'entropy']
}

# 4 métricas de validación
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# GridSearchCV con 4-fold estratificado (como en los notebooks del curso)
cv_strategy = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)

print("\nEjecutando GridSearchCV para Árbol de Decisión...")
print(f"Parámetros: {dt_param_grid}")
print(f"Métricas de validación: {scoring_metrics}")
print(f"CV: StratifiedKFold(n_splits=4)")

dt_grid_search = GridSearchCV(
    estimator=dt_base,
    param_grid=dt_param_grid,
    cv=cv_strategy,
    scoring=scoring_metrics,
    refit='accuracy',
    return_train_score=True,
    n_jobs=-1
)
dt_grid_search.fit(X_train, y_train)

# Analizar resultados
dt_results = pd.DataFrame(dt_grid_search.cv_results_)

print(f"\nTotal combinaciones evaluadas: {len(dt_results)}")
print(f"\nMejor modelo (por Accuracy):")
print(f"  Params: {dt_grid_search.best_params_}")
print(f"  Accuracy CV media: {dt_grid_search.best_score_:.4f}")

# Mostrar top 5 modelos por cada métrica
for metric in scoring_metrics:
    col = f'mean_test_{metric}'
    top5 = dt_results.nlargest(5, col)[['params', col, f'std_test_{metric}']].reset_index(drop=True)
    print(f"\nTop 5 modelos por {metric}:")
    print(top5.to_string())

# Boxplots de los top 5 candidatos por Accuracy para evaluar robustez
dt_sorted = dt_results.nlargest(10, 'mean_test_accuracy').reset_index(drop=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, metric in enumerate(scoring_metrics):
    ax = axes[idx // 2][idx % 2]
    split_cols = [f'split{i}_test_{metric}' for i in range(4)]
    data_to_plot = []
    labels = []
    for i in range(min(5, len(dt_sorted))):
        data_to_plot.append(dt_sorted[split_cols].iloc[i].values)
        labels.append(f"M{i+1}")
    ax.boxplot(data_to_plot, labels=labels)
    ax.set_title(f'Top 5 modelos - {metric}')
    ax.set_ylabel(metric)
plt.suptitle('Árbol de Decisión: Robustez de los mejores candidatos', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig04_dt_boxplots_robustez.png'), dpi=150)
plt.close()

# Seleccionar el mejor árbol
best_dt = dt_grid_search.best_estimator_
print(f"\nMejor Árbol seleccionado: {best_dt}")

# Predicciones train y test para detectar sobreajuste
y_pred_train_dt = best_dt.predict(X_train)
y_pred_test_dt = best_dt.predict(X_test)

acc_train_dt = accuracy_score(y_train, y_pred_train_dt)
acc_test_dt = accuracy_score(y_test, y_pred_test_dt)
print(f"\nAccuracy Train: {acc_train_dt:.4f}")
print(f"Accuracy Test:  {acc_test_dt:.4f}")
if acc_train_dt - acc_test_dt > 0.05:
    print("AVISO: Posible sobreajuste (diferencia > 5%)")

print(f"\nClassification Report (Test):\n{classification_report(y_test, y_pred_test_dt)}")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm_dt = confusion_matrix(y_test, y_pred_test_dt)
ConfusionMatrixDisplay(cm_dt, display_labels=['Black', 'White']).plot(ax=ax, cmap='Blues')
ax.set_title('Matriz de Confusión - Mejor Árbol de Decisión')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig05_dt_confusion_matrix.png'), dpi=150)
plt.close()

# ROC Curve
y_prob_dt = best_dt.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
auc_dt = auc(fpr_dt, tpr_dt)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr_dt, tpr_dt, label=f'Árbol de Decisión (AUC={auc_dt:.4f})', color='blue')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Curva ROC - Mejor Árbol de Decisión')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig06_dt_roc_curve.png'), dpi=150)
plt.close()

# Representación gráfica del árbol
fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(best_dt, ax=ax, filled=True, proportion=True,
          feature_names=X.columns.tolist(),
          class_names=['Black', 'White'],
          fontsize=8, rounded=True)
ax.set_title('Mejor Árbol de Decisión', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig07_dt_tree_plot.png'), dpi=150, bbox_inches='tight')
plt.close()

# Reglas en formato texto
tree_rules = export_text(best_dt, feature_names=X.columns.tolist())
print(f"\nReglas del Árbol de Decisión (formato texto):")
print(tree_rules)

# Importancia de variables
fig, ax = plt.subplots(figsize=(10, 6))
importance_dt = pd.Series(best_dt.feature_importances_, index=X.columns)
importance_dt_sorted = importance_dt.sort_values(ascending=True)
importance_dt_sorted[importance_dt_sorted > 0].plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Importancia de Variables - Árbol de Decisión')
ax.set_xlabel('Importancia')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig08_dt_feature_importance.png'), dpi=150)
plt.close()

print("\nImportancia de variables:")
print(importance_dt.sort_values(ascending=False))

# =============================================================================
# 3) BÚSQUEDA PARAMÉTRICA - RANDOM FOREST Y XGBOOST (según Accuracy)
# =============================================================================
print("\n" + "=" * 80)
print("3) BÚSQUEDA PARAMÉTRICA - RANDOM FOREST Y XGBOOST")
print("=" * 80)

# ---- 3.1 RANDOM FOREST ----
print("\n--- 3.1 RANDOM FOREST ---")
rf_base = RandomForestClassifier(random_state=SEED)

rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 10, 15, 20],
    'min_samples_split': [5, 10, 20, 50],
    'min_samples_leaf': [3, 10, 30],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True]
}

print(f"Parámetros RF: {rf_param_grid}")
print("Ejecutando GridSearchCV para Random Forest...")

rf_grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=rf_param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    refit=True,
    return_train_score=True,
    n_jobs=-1
)
rf_grid_search.fit(X_train, y_train)

rf_results = pd.DataFrame(rf_grid_search.cv_results_)

print(f"\nTotal combinaciones evaluadas: {len(rf_results)}")
print(f"\nMejor modelo Random Forest:")
print(f"  Params: {rf_grid_search.best_params_}")
print(f"  Accuracy CV media: {rf_grid_search.best_score_:.4f}")

# Top 5 RF por Accuracy
rf_top5 = rf_results.nlargest(5, 'mean_test_score')[
    ['params', 'mean_test_score', 'std_test_score']
].reset_index(drop=True)
print(f"\nTop 5 Random Forest por Accuracy:")
print(rf_top5.to_string())

best_rf = rf_grid_search.best_estimator_

# Predicciones
y_pred_train_rf = best_rf.predict(X_train)
y_pred_test_rf = best_rf.predict(X_test)
acc_train_rf = accuracy_score(y_train, y_pred_train_rf)
acc_test_rf = accuracy_score(y_test, y_pred_test_rf)

print(f"\nAccuracy Train: {acc_train_rf:.4f}")
print(f"Accuracy Test:  {acc_test_rf:.4f}")
if acc_train_rf - acc_test_rf > 0.05:
    print("AVISO: Posible sobreajuste (diferencia > 5%)")

print(f"\nClassification Report (Test):\n{classification_report(y_test, y_pred_test_rf)}")

# Boxplots de robustez RF
fig, ax = plt.subplots(figsize=(8, 5))
rf_sorted = rf_results.nlargest(5, 'mean_test_score').reset_index(drop=True)
split_cols_rf = [f'split{i}_test_score' for i in range(4)]
data_rf = [rf_sorted[split_cols_rf].iloc[i].values for i in range(min(5, len(rf_sorted)))]
ax.boxplot(data_rf, labels=[f"M{i+1}" for i in range(len(data_rf))])
ax.set_title('Random Forest: Robustez Top 5 candidatos (Accuracy)')
ax.set_ylabel('Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig09_rf_boxplots_robustez.png'), dpi=150)
plt.close()

# Feature importance RF
fig, ax = plt.subplots(figsize=(10, 6))
importance_rf = pd.Series(best_rf.feature_importances_, index=X.columns)
importance_rf_sorted = importance_rf.sort_values(ascending=True)
importance_rf_sorted[importance_rf_sorted > 0].plot(kind='barh', ax=ax, color='forestgreen')
ax.set_title('Importancia de Variables - Random Forest')
ax.set_xlabel('Importancia')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig10_rf_feature_importance.png'), dpi=150)
plt.close()

# ---- 3.2 XGBOOST ----
print("\n--- 3.2 XGBOOST ---")
xgb_base = XGBClassifier(
    booster='gbtree',
    tree_method='hist',
    random_state=SEED,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'gamma': [0, 0.1, 0.5],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.8, 1.0]
}

print(f"Parámetros XGBoost: {xgb_param_grid}")
print("Ejecutando GridSearchCV para XGBoost...")

xgb_grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=xgb_param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    refit=True,
    return_train_score=True,
    n_jobs=-1
)
xgb_grid_search.fit(X_train, y_train)

xgb_results = pd.DataFrame(xgb_grid_search.cv_results_)

print(f"\nTotal combinaciones evaluadas: {len(xgb_results)}")
print(f"\nMejor modelo XGBoost:")
print(f"  Params: {xgb_grid_search.best_params_}")
print(f"  Accuracy CV media: {xgb_grid_search.best_score_:.4f}")

# Top 5 XGBoost por Accuracy
xgb_top5 = xgb_results.nlargest(5, 'mean_test_score')[
    ['params', 'mean_test_score', 'std_test_score']
].reset_index(drop=True)
print(f"\nTop 5 XGBoost por Accuracy:")
print(xgb_top5.to_string())

best_xgb = xgb_grid_search.best_estimator_

# Predicciones
y_pred_train_xgb = best_xgb.predict(X_train)
y_pred_test_xgb = best_xgb.predict(X_test)
acc_train_xgb = accuracy_score(y_train, y_pred_train_xgb)
acc_test_xgb = accuracy_score(y_test, y_pred_test_xgb)

print(f"\nAccuracy Train: {acc_train_xgb:.4f}")
print(f"Accuracy Test:  {acc_test_xgb:.4f}")
if acc_train_xgb - acc_test_xgb > 0.05:
    print("AVISO: Posible sobreajuste (diferencia > 5%)")

print(f"\nClassification Report (Test):\n{classification_report(y_test, y_pred_test_xgb)}")

# Boxplots de robustez XGBoost
fig, ax = plt.subplots(figsize=(8, 5))
xgb_sorted = xgb_results.nlargest(5, 'mean_test_score').reset_index(drop=True)
split_cols_xgb = [f'split{i}_test_score' for i in range(4)]
data_xgb = [xgb_sorted[split_cols_xgb].iloc[i].values for i in range(min(5, len(xgb_sorted)))]
ax.boxplot(data_xgb, labels=[f"M{i+1}" for i in range(len(data_xgb))])
ax.set_title('XGBoost: Robustez Top 5 candidatos (Accuracy)')
ax.set_ylabel('Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig11_xgb_boxplots_robustez.png'), dpi=150)
plt.close()

# Feature importance XGBoost
fig, ax = plt.subplots(figsize=(10, 6))
importance_xgb = pd.Series(best_xgb.feature_importances_, index=X.columns)
importance_xgb_sorted = importance_xgb.sort_values(ascending=True)
importance_xgb_sorted[importance_xgb_sorted > 0].plot(kind='barh', ax=ax, color='darkorange')
ax.set_title('Importancia de Variables - XGBoost')
ax.set_xlabel('Importancia')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig12_xgb_feature_importance.png'), dpi=150)
plt.close()


# =============================================================================
# 4) COMPARACIÓN COMPLETA DE MODELOS
# =============================================================================
print("\n" + "=" * 80)
print("4) COMPARACIÓN COMPLETA DE MODELOS")
print("=" * 80)

# 4.1 Tabla resumen de resultados
models_summary = {
    'Modelo': ['Árbol de Decisión', 'Random Forest', 'XGBoost'],
    'Acc Train': [acc_train_dt, acc_train_rf, acc_train_xgb],
    'Acc Test': [acc_test_dt, acc_test_rf, acc_test_xgb],
    'Diff (Train-Test)': [
        acc_train_dt - acc_test_dt,
        acc_train_rf - acc_test_rf,
        acc_train_xgb - acc_test_xgb
    ],
    'Best CV Accuracy': [
        dt_grid_search.best_score_,
        rf_grid_search.best_score_,
        xgb_grid_search.best_score_
    ]
}

# Precision, Recall, F1 en test
for name, y_pred in zip(
    ['Árbol de Decisión', 'Random Forest', 'XGBoost'],
    [y_pred_test_dt, y_pred_test_rf, y_pred_test_xgb]
):
    pass  # Already shown in classification reports

summary_df = pd.DataFrame(models_summary)
print(f"\nResumen comparativo:")
print(summary_df.to_string(index=False))

# 4.2 Métricas detalladas por modelo en test
print("\n--- Métricas detalladas en Test ---")
detailed_metrics = []
for name, y_pred, model in zip(
    ['Árbol de Decisión', 'Random Forest', 'XGBoost'],
    [y_pred_test_dt, y_pred_test_rf, y_pred_test_xgb],
    [best_dt, best_rf, best_xgb]
):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc_val = roc_auc_score(y_test, y_prob)
    detailed_metrics.append({
        'Modelo': name, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1-Score': f1, 'AUC': auc_val
    })

detail_df = pd.DataFrame(detailed_metrics)
print(detail_df.to_string(index=False))

# 4.3 Comparación con validación cruzada repetida (RepeatedStratifiedKFold)
print("\n--- Validación Cruzada Repetida (5-fold x 10 repeats) ---")
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
cv_results_all = {}
for name, model in zip(
    ['Árbol de Decisión', 'Random Forest', 'XGBoost'],
    [best_dt, best_rf, best_xgb]
):
    scores = cross_val_score(model, X_train, y_train, cv=rcv, scoring='accuracy', n_jobs=-1)
    cv_results_all[name] = scores
    print(f"{name}: Accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

# 4.4 Boxplots comparativos de CV
fig, ax = plt.subplots(figsize=(10, 6))
boxplot_data = [cv_results_all[name] for name in cv_results_all]
bp = ax.boxplot(boxplot_data, labels=list(cv_results_all.keys()), patch_artist=True)
colors = ['#3498db', '#2ecc71', '#e67e22']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Comparación de Modelos - Accuracy (Repeated Stratified 5-Fold CV x10)')
ax.set_ylabel('Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig13_comparacion_boxplots.png'), dpi=150)
plt.close()

# 4.5 Comparación de curvas ROC
fig, ax = plt.subplots(figsize=(8, 6))
for name, model, color in zip(
    ['Árbol de Decisión', 'Random Forest', 'XGBoost'],
    [best_dt, best_rf, best_xgb],
    ['blue', 'green', 'orange']
):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})', color=color)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Curvas ROC - Comparación de Modelos')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig14_comparacion_roc.png'), dpi=150)
plt.close()

# 4.6 Confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, name, y_pred in zip(
    axes,
    ['Árbol de Decisión', 'Random Forest', 'XGBoost'],
    [y_pred_test_dt, y_pred_test_rf, y_pred_test_xgb]
):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Black', 'White']).plot(ax=ax, cmap='Blues')
    ax.set_title(name)
plt.suptitle('Matrices de Confusión - Comparación de Modelos', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig15_comparacion_confusion_matrices.png'), dpi=150)
plt.close()

# 4.7 Bar chart de accuracy en test
fig, ax = plt.subplots(figsize=(8, 5))
model_names = ['Árbol de Decisión', 'Random Forest', 'XGBoost']
acc_values = [acc_test_dt, acc_test_rf, acc_test_xgb]
bars = ax.bar(model_names, acc_values, color=colors, alpha=0.8, edgecolor='black')
for bar, val in zip(bars, acc_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Accuracy en Test - Comparación de Modelos')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.4, max(acc_values) + 0.05)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig16_comparacion_accuracy_bar.png'), dpi=150)
plt.close()

# 4.8 Feature importance comparativa
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for ax, name, importance, color in zip(
    axes,
    ['Árbol de Decisión', 'Random Forest', 'XGBoost'],
    [importance_dt, importance_rf, importance_xgb],
    ['steelblue', 'forestgreen', 'darkorange']
):
    imp_sorted = importance.sort_values(ascending=True)
    imp_sorted[imp_sorted > 0].plot(kind='barh', ax=ax, color=color)
    ax.set_title(f'Importancia - {name}')
    ax.set_xlabel('Importancia')
plt.suptitle('Comparación de Importancia de Variables', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig17_comparacion_feature_importance.png'), dpi=150)
plt.close()

print("\nTodas las figuras guardadas en 'figuras/'")
print("\n" + "=" * 80)
print("ANÁLISIS COMPLETADO")
print("=" * 80)
