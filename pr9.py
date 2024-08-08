import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
def kernel(point, xmat, k):
 m, n = xmat.shape
 weights = np.mat(np.eye(m))
 for j in range(m):
  diff = point - xmat[j]
  weights[j, j] = np.exp(diff.dot(diff.T) / (-2.0 * k**2))
 return weights
def local_weight(point, xmat, ymat, k):
 wei = kernel(point, xmat, k)
 mata = xmat.T * wei * xmat
 W = np.linalg.pinv(mata).dot(xmat.T * wei * ymat)
 return W
def local_weight_regression(xmat, ymat, k):
 m, n = xmat.shape
 ypred = np.zeros(m)
 for i in range(m):
  ypred[i] = xmat[i].dot(local_weight(xmat[i], xmat, ymat, k))
 return ypred
data = pd.read_csv('tips.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
mbill = bill.reshape(-1, 1)
one = np.ones((mbill.shape[0], 1))
X = np.hstack((one, mbill))
ypred = local_weight_regression(X, tip.reshape(-1, 1), 0.3)
SortIndex = X[:, 1].argsort()
xsort = X[SortIndex]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color='green')
ax.plot(xsort[:, 1], ypred[SortIndex], color='red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()
