import numpy as np

eps = 10 ** - 14
A = [[2.32, 0.12, 1.57], [1.69, 4.17, -0.33], [0.45, 1.82, 3.15]]
b = [2.01, 3.23, 0.94]

print(np.linalg.solve(A, b))
print()

x = [np.ones(len(b))]
r = [b - np.dot(A, x[0])]
z = [r[0]]

for k in range(1, 10 ** 10):
    alpha = sum(np.multiply(r[k - 1], r[k - 1])) / sum(np.multiply(np.dot(A, z[k - 1]), z[k - 1]))
    r.append(r[k - 1] - np.dot(np.dot(alpha, A), z[k - 1]))
    x.append(x[k - 1] + np.dot(alpha, z[k - 1]))
    beta = sum(np.multiply(r[k], r[k])) / sum(np.multiply(r[k - 1], r[k - 1]))
    z.append(r[k] - np.dot(beta, z[k - 1]))

    if sum(map(abs, x[-2] - x[-1])) < eps:
        break

print("Made %d iterations" % k)

print(x[-1])
