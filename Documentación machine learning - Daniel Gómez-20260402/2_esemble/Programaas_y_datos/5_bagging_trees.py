# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:19:37 2023

@author: user ## analizando en bagging trees el efecto del numero de clasificadores y tamaño de la muestra
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
import matplotlib.pylab as plt
from matplotlib.ticker import StrMethodFormatter

wine_pd = pd.read_csv('D:/documentos_en_D/docencia_master/SVM_emsemble_2024_dani/2_Introduccion_y_SVM (1)/2_Introduccion_y_SVM/Wine.csv')
wine_pd.head()

y = wine_pd.pop('Customer_Segment').values
X = wine_pd.values
X.shape

dtree = DecisionTreeClassifier(criterion='entropy')

## como afecta el numero de replicas al bagging. Ejemplo de arboles

n_reps = 50 ## repeticiones para que no dependa de la semilla  
folds = 4
est_range = range(2,16)
n_est_dict = {}
for n_est in est_range: 
    scores = []
    for rep in range(n_reps):
        tree_bag = BaggingClassifier(dtree, 
                            n_estimators = n_est,
                            max_samples = 1.0, # bootstrap resampling 
                            bootstrap = True)
        scores_tree_bag = cross_val_score(tree_bag, X, y, cv=folds, n_jobs = -1)
        scores.append(scores_tree_bag.mean())
    n_est_dict[n_est]=np.array(scores).mean()
    
res_list = sorted(n_est_dict.items()) # sorted by key, return a list of tuples
nc, accs = zip(*res_list) # unpack a list of pairs into two tuples
f = plt.figure(figsize=(5,4))

plt.plot(nc, accs, lw = 3)

plt.xlabel("Number of clasificadores")
plt.ylabel("Accuracy")
f.savefig('acc-est.pdf')


n_reps = 50
n_est = 10
res_dict = {}
max_s_range = np.arange(0.95,0.4,-0.05)
for max_s in max_s_range:
    scores = []
    for rep in range(n_reps):
        tree_bag = BaggingClassifier(dtree, 
                            n_estimators = n_est,
                            max_samples = max_s,  
                            bootstrap = False)
        scores_tree_bag = cross_val_score(tree_bag, X, y, cv=folds, n_jobs = -1)
        scores.append(scores_tree_bag.mean())
    res_dict[max_s]=np.array(scores).mean()
    

res_list = sorted(res_dict.items()) # sorted by key, return a list of tuples
ns, accs = zip(*res_list) # unpack a list of pairs into two tuples
f = plt.figure(figsize=(5,4))

plt.plot(ns, accs, lw = 3, color = 'r')
plt.xlim([1, 0.4])
#plt.ylim([0.88, 0.92])
plt.xlabel("Proporción de datos usados en el muestreo. Tamaño de la submuestras con 10 arboles ")
plt.ylabel("Accuracy")
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
f.savefig('acc-div.pdf')
#########

n_reps = 50
n_est = 20
res_dict = {}
max_s_range = np.arange(0.95,0.4,-0.05)
for max_s in max_s_range:
    scores = []
    for rep in range(n_reps):
        tree_bag = BaggingClassifier(dtree, 
                            n_estimators = n_est,
                            max_samples = max_s,  
                            bootstrap = False)
        scores_tree_bag = cross_val_score(tree_bag, X, y, cv=folds, n_jobs = -1)
        scores.append(scores_tree_bag.mean())
    res_dict[max_s]=np.array(scores).mean()
    

res_list = sorted(res_dict.items()) # sorted by key, return a list of tuples
ns, accs = zip(*res_list) # unpack a list of pairs into two tuples
f = plt.figure(figsize=(5,4))

plt.plot(ns, accs, lw = 3, color = 'r')
plt.xlim([1, 0.4])
#plt.ylim([0.88, 0.92])
plt.xlabel("Tamaño de la submuestras con 20 arboles ")
plt.ylabel("Accuracy")
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
f.savefig('acc-div.pdf')






############ otro ejemplo si no decimos nada estamos con un treeee

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

data = datasets.load_wine(as_frame = True)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)

estimator_range = [2,4,6,8,10,12,14,16]

models = []
scores = []

for n_estimators in estimator_range:

    # Create bagging classifier
    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)

    # Fit the model
    clf.fit(X_train, y_train)

    # Append the model and score to their respective list
    models.append(clf)
    scores.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))

# Generate the plot of scores against number of estimators
plt.figure(figsize=(9,6))
plt.plot(estimator_range, scores)

# Adjust labels and font (to make visable)
plt.xlabel("n_estimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize = 16)

# Visualize plot
plt.show()


## podemos ver que pasa
from sklearn.tree import plot_tree
plt.figure(figsize=(30, 20))

plot_tree(clf.estimators_[0], feature_names = X.columns)



