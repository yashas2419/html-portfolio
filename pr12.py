import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)
y = np.sin(x) + 0.1 * np.random.randn(100)
x_new = np.linspace(-10, 10, 1000)
y_pred = np.zeros(1000)
for i in range(1000):
    weights = np.exp(-((x - x_new[i]) / 1) ** 2)
    y_pred[i] = np.sum(weights * y) / np.sum(weights)
plt.scatter(x, y, label='Data')
plt.plot(x_new[:-1], y_pred[:-1], label='Locally Weighted Regression')
plt.legend()
plt.show()