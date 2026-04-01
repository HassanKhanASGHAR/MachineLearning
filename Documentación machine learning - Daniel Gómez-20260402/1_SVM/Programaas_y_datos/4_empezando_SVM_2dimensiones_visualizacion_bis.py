# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 12:13:06 2023

@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import svm

X1 = pd.DataFrame([1, 2, 3, 5, 6 ,7])
X2= pd.DataFrame([4, 3, 5, 2, 1, 3])

X1=X1.set_axis(['X1'], axis=1)
X2=X2.set_axis(['X2'], axis=1)

X= pd.concat([X1,X2], axis=1)          
     
y = pd.DataFrame([-1, -1, -1, 1, 1, 1])
y=y.set_axis(['y'], axis=1)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=np.ravel(y), s=50, cmap='autumn')


model = svm.SVC(kernel='linear', C=1)
model.fit(X, np.ravel(y))

# get support vectors
model.support_vectors_

# get indices of support vectors
model.support_

# get number of support vectors for each class
model.n_support_

model.decision_function

# Obtén los coeficientes (vectores de peso)
weights = model.coef_
bias = model.intercept_

# Imprime los resultados
print("Coeficientes del Hiperplano:", weights)
print("Coeficiente b del hiperplano:", bias)

####################
# Dibujar la nube de puntos
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=np.ravel(y), s=50, cmap='autumn')

# Dibujar el hiperplano
slope = -weights[0, 0] / weights[0, 1]  # Calcula la pendiente del hiperplano
intercept = -bias / weights[0, 1]  # Calcula el término de intercepción del hiperplano

# Dibujar la línea que representa el hiperplano
ax = plt.gca()
xlim = ax.get_xlim()
xx = np.linspace(xlim[0], xlim[1])
yy = slope * xx + intercept
plt.plot(xx, yy, 'k-', label='Hiperplano Separador')

# Resaltar los vectores de soporte
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=200, facecolors='none', edgecolors='k', marker='o', label='Vectores de Soporte')

plt.legend()
plt.show()



### visualizando SVM
import seaborn as sns
iris = sns.load_dataset("iris")
print(iris.head())
y = iris.species
X = iris.drop('species',axis=1)
sns.pairplot(iris, hue="species",palette="bright")

## para simplificar el problema quitamos la clase virginica y las variables sepal.
## Asi tenemos un problema de clasificacion de dos clases con dos variables que podemos visualizar


df=iris[(iris['species']!='virginica')]
df=df.drop(['sepal_length','sepal_width'], axis=1)
df.head()

sns.pairplot(df, hue="species",palette="bright")


#Cambiamos valores categoricos en númericos 
df=df.replace('setosa', 0)
df=df.replace('versicolor', 1)
X=df.iloc[:,0:2]  ## para coger las dos primeras columnas
V=df[['petal_length', 'petal_width']] ## otra manera de coger las dos columnas


y=df['species']
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
plt.scatter(df[['petal_length']], df[['petal_width']] , c=y, s=50, cmap='autumn') ## con la V


### dibujamos iris
from sklearn.svm import SVC 
model = SVC(kernel='linear', C=2)
model.fit(X, y)
model.support_vectors_  ### vectores soporte del problema sobre 

# Obtén los coeficientes (vectores de peso)
weights = model.coef_
bias = model.intercept_

# Imprime los resultados
print("Coeficientes del Hiperplano:", weights)
print("Coeficiente b del hiperplano:", bias)

y=df['species']
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
plt.scatter(df[['petal_length']], df[['petal_width']] , c=y, s=50, cmap='autumn') ## con la V


# Dibujar el hiperplano
slope = -weights[0, 0] / weights[0, 1]  # Calcula la pendiente del hiperplano
intercept = -bias / weights[0, 1]  # Calcula el término de intercepción del hiperplano

# Dibujar la línea que representa el hiperplano
ax = plt.gca()
xlim = ax.get_xlim()
xx = np.linspace(xlim[0], xlim[1])
yy = slope * xx + intercept
plt.plot(xx, yy, 'k-', label='Hiperplano Separador')

# Resaltar los vectores de soporte
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=200, facecolors='none', edgecolors='k', marker='o', label='Vectores de Soporte')

plt.legend()
plt.show()






#######################################
########### VISUALIZACION NUMERO 1

## los pintamos para visualizarlos
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1])
### hiperplano separador

ax = plt.gca()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
xlim = ax.get_xlim() ## limites de la variable x
ylim = ax.get_ylim() ## ## limites de la variable y

xx = np.linspace(xlim[0], xlim[1], 30) ## dividimos el espacio de la x en 30 partes iguales
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()

############### VISUALIZACION NUMERO 2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')

### pintamos hiperplanos a mano

xfit = np.linspace(1, 6)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
plt.plot([2.6], [0.9], 'x', color='red', markeredgewidth=2, markersize=10) # el punto que aparece con la x

#for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
#    plt.plot(xfit, m * xfit + b, '-k')

plt.xlim(1, 5.5);

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);

model.support_vectors_

#### SOFT MARGIN, el efecto de el parametro C

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0, cluster_std=1.2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

model=SVC(kernel='linear', C=0.1)
clf = model.fit(X, y)

plot_svc_decision_function(clf, plot_support=False);

###### SOFT COST 
X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=0.8)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none');
    axi.set_title('C = {0:.1f}'.format(C), size=14)



## pipeline

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC())
