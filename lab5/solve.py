import numpy as np
import matplotlib.pyplot as plt
import math


def make_A(xs):
    N = len(xs)
    A = np.zeros((N, N))

    # for n in range(N):
    #     for k in range(N):
    #         if abs(i - j) == 1:
                # A[i][j] = 1 / (xs[i] - xs[j])

    return A


def make_B(xs):
    b = []

    first = lambda a, b: -(a - b) * (a ** 2 + 2 * a * b + 3 * b ** 2) / 12
    second = lambda a, b: -(a - b) * (3 * a ** 2 + 2 * a * b + b ** 2) / 12

    for i in range(len(xs)):
        bi = 0
        if i > 0:
            bi += first(xs[i - 1], xs[i])
        if i < len(xs) - 1:
            bi += second(xs[i], xs[i + 1])
        b.append(bi)

    return b


a = 0
b = 1


def true_f(x):
    return 2 * math.tan(1 / 2) * math.sin(x) + 2 * math.cos(x) + x ** 2 - 2


def g(x):
    return x ** 2


# ns = [x * 100 + 5 for x in range(16)]
ns = [20]

es = []
deltas = []
for N in ns:

    delta = (b - a) / N
    xs = [a + x * delta for x in range(N + 1)]

    B = make_B(xs)
    A = make_A(xs)

    f = np.linalg.solve(A, B)
    f = list(f)

    print(f)

    plt.plot(xs, f)
    plt.show()

    break

    errors = []
    for k in range(1, N):
        x = (a + k * delta)
        expected = true_f(x)
        errors.append(abs(expected - f[k]) / expected)

    es.append(max(errors))
    deltas.append(1 / N)

# plt.yscale('log')
# plt.xscale('log')
# plt.plot(deltas, es)
# plt.show()
