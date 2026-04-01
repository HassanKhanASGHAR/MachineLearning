"""
===================================================================
Support Vector Regression (SVR) using linear and non-linear kernels
===================================================================

Toy example of 1D regression using linear, polynomial and RBF kernels.

"""
print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

###############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

###############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

###############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=10, gamma=0.1)
svr_lin = SVR(kernel='linear', C=0.5)
svr_poly = SVR(kernel='poly', C=0.5, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

###############################################################################
# look at the results

plt.figure(figsize=(15, 15))

lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

#################
# Calculate errors
mse_rbf = mean_squared_error(y, y_rbf)
r2_rbf = r2_score(y, y_rbf)

mse_lin = mean_squared_error(y, y_lin)
r2_lin = r2_score(y, y_lin)

mse_poly = mean_squared_error(y, y_poly)
r2_poly = r2_score(y, y_poly)

# Print errors
print("RBF Model:")
print("MSE: {:.4f}".format(mse_rbf))
print("R2 Score: {:.4f}".format(r2_rbf))
print()

print("Linear Model:")
print("MSE: {:.4f}".format(mse_lin))
print("R2 Score: {:.4f}".format(r2_lin))
print()

print("Polynomial Model:")
print("MSE: {:.4f}".format(mse_poly))
print("R2 Score: {:.4f}".format(r2_poly))
