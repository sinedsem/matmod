import numpy as np
import matplotlib.pyplot as plt
import math


def get_g(a, fa, b, fb, n, g):
    delta = (b - a) / n

    xs = [a + delta * x for x in range(n - 1)]
    result = list(map(g, xs))
    result[0] -= fa / (delta ** 2)
    result[-1] -= fb / (delta ** 2)

    return delta, result, xs


def make_A(k, delta):
    k -= 1
    delta2 = delta ** 2
    A = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                A[i][j] = -2 / delta2 + 1
            elif abs(i - j) == 1:
                A[i][j] = 1 / delta2
    return A


a = 0

g = lambda x: 2 * math.tan(1 / 2) * math.sin(x) + 2 * math.cos(x) + x ** 2 - 2

es = []
ns = []
for N in range(5, 1606, 100):
    delta, gs, xs = get_g(a, 0, 1, 1, N, lambda x: x ** 2)
    A = make_A(N, delta)
    print(len(gs))
    f = np.linalg.solve(A, gs)
    f = list(f)

    xs.insert(0, 0)
    xs.append(1)
    f.insert(0, 0)
    f.append(1)

    errors = []
    for k in range(1, N):
        x = (a + k * delta)
        expected = g(x)
        errors.append(abs(expected - f[k]) / expected)

    es.append(max(errors))
    ns.append(1 / N)

plt.yscale('log')
plt.xscale('log')
plt.plot(ns, es)
plt.show()
