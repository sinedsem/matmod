import numpy as np
import matplotlib.pyplot as plt

eps = 10 ** - 14
# A = [[3, 4, 0], [4, -3, 0], [0, 0, 5]]
A = [[3, 4.1, 0], [4, -3.1, 0], [0, 0, 5.1]]

true_x = np.array([2, 1, 4])

b = np.dot(A, true_x)

x = [np.zeros(len(b))]
r = [b - np.dot(A, x[0])]
z = [r[0]]

residual = []
error = []
for k in range(1, 10 ** 3):
    alpha = sum(np.multiply(r[k - 1], r[k - 1])) / sum(np.multiply(np.dot(A, z[k - 1]), z[k - 1]))
    r.append(r[k - 1] - np.dot(np.dot(alpha, A), z[k - 1]))
    x.append(x[k - 1] + np.dot(alpha, z[k - 1]))
    beta = sum(np.multiply(r[k], r[k])) / sum(np.multiply(r[k - 1], r[k - 1]))
    z.append(r[k] + np.dot(beta, z[k - 1]))

    residual.append(np.sqrt(sum((np.dot(A, x[-1]) - b) ** 2)))
    error.append(np.sqrt(sum((x[-1] - true_x) ** 2)))

    if error[-1] < eps:
        break

print("Made %d iterations" % k)

plt.plot(range(1, len(residual) + 1), residual)
plt.plot(range(1, len(error) + 1), error)
plt.yscale('log')
plt.show()

print(x[-1])
