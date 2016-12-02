import numpy as np
import matplotlib.pyplot as plt
import math


def make_A(xs, delta):
    N = len(xs)
    N -= 2
    A = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                A[i][j] = -2 / delta
            elif abs(i - j) == 1:
                A[i][j] = 1 / delta

    return A


def make_B(xs):
    b = []

    first = lambda a, b: -(a - b) * (a ** 2 + 2 * a * b + 3 * b ** 2) / 12
    second = lambda a, b: -(a - b) * (3 * a ** 2 + 2 * a * b + b ** 2) / 12

    for i in range(len(xs) - 2):
        bi = 0
        if i > 0:
            bi += first(xs[i - 1], xs[i])
        if i < len(xs) - 1:
            bi += second(xs[i], xs[i + 1])
        b.append(bi)

    b[0] -= 0
    b[-1] -= 1 / delta

    return b


a = 0
b = 1


def true_f(x):
    return x * (11 + x ** 3) / 12


def g(x):
    return x ** 2


ns = [x * 100 + 5 for x in range(16)]
# ns = [100]

es = []
deltas = []
for N in ns:

    delta = (b - a) / N
    xs = [a + x * delta for x in range(N + 1)]

    B = make_B(xs)

    A = make_A(xs, delta)

    f = np.linalg.solve(A, B)
    f = list(f)

    f.insert(0, 0)
    f.append(1)
    # plt.plot(xs, f)
    # plt.plot(xs, list(map(true_f, xs)))
    # plt.show()

    errors = []
    for k in range(1, N):
        x = (a + k * delta)
        expected = true_f(x)
        errors.append((expected - f[k]) ** 2)

    es.append(sum(errors))
    deltas.append(1 / N)

plt.yscale('log')
plt.xscale('log')
plt.plot(deltas, es)
plt.show()
