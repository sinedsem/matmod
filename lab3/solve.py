import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
from scipy.sparse import lil_matrix, csr_matrix

from cgm import solve_cgm, quick_solve_cgm

eps = 10 ** -5


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def get_B(a, b, N):
    result = np.zeros((N, N))

    delta = (b - a) / N
    delta2 = delta ** 2
    delta2n = delta ** -2

    xs = [a + delta * x for x in range(N)]
    ys = [a + delta * x for x in range(N)]

    for k in range(N):
        for j in range(N):
            if (k - N / 2) ** 2 + (j - N / 2) ** 2 <= delta2n:
                result[k][j] = delta2

    return delta, result, xs, ys


def make_A_2d(N, delta):
    half_N = N / 2
    delta2n = delta ** -2
    A = lil_matrix((N ** 2, N ** 2))

    for k in range(N):
        for j in range(N):
            # print(k * N + k, j * N + j)
            A[k * N + j, k * N + j] = -4

    for k in range(1, N - 1):
        for j in range(1, N - 1):
            A[k * N + j, (k + 1) * N + j] = 1
            A[k * N + j, (k - 1) * N + j] = 1
            A[k * N + j, k * N + j + 1] = 1
            A[k * N + j, k * N + j - 1] = 1

            A[(k + 1) * N + j, k * N + j] = 1
            A[(k - 1) * N + j, k * N + j] = 1
            A[k * N + j + 1, k * N + j] = 1
            A[k * N + j - 1, k * N + j] = 1

    for k in range(1, N - 1):
        for j in range(1, N - 1):
            if (k - half_N) ** 2 + (j - half_N) ** 2 > delta2n:
                A[k * N + j, (k + 1) * N + j] = 0
                A[k * N + j, (k - 1) * N + j] = 0
                A[k * N + j, k * N + j + 1] = 0
                A[k * N + j, k * N + j - 1] = 0

                A[(k + 1) * N + j, k * N + j] = 0
                A[(k - 1) * N + j, k * N + j] = 0
                A[k * N + j + 1, k * N + j] = 0
                A[k * N + j - 1, k * N + j] = 0

    # for j in range(N ** 2):
    #     for k in range(N ** 2):
    #         if abs(A[k, j] - A[j, k]) == 1:
    #             A[k, j] = 0
    #             A[j, k] = 0

    return A


def draw_plot(xs, ys, u):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(xs, ys)
    # x = y = np.arange(-3.0, 3.0, 0.05)
    # X, Y = np.meshgrid(x, y)
    # zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    # Z = zs.reshape(X.shape)
    #
    surf = ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


a = -1
b = 1

g = lambda x, y: (x ** 2 + y ** 2 - 1) / 4 if x ** 2 + y ** 2 < 1 else 0

# def g(x, y):
#     if x ** 2 + y ** 2 < 1:
#         return (x ** 2 + y ** 2 - 1) / 4
#     else:
#         return 0


es = []
residuals = []
deltas = []
# ns = [10, 15, 20, 50, 100, 200, 400, 800, 1500]
# ns = [10, 15, 20, 40, 50, 75, 100]
ns = [100]

for N in ns:

    delta, B, xs, ys = get_B(a, b, N)

    A_reshaped = make_A_2d(N, delta)
    B_reshaped = np.reshape(B, (N ** 2))

    # print(abs((A_reshaped - A_reshaped.transpose())).max())
    print(A_reshaped.toarray())
    # print(A_reshaped)
    # break

    u_reshaped = quick_solve_cgm(A_reshaped, B_reshaped, eps)

    u = np.reshape(u_reshaped, (N, N))

    # draw_plot(xs, ys, u)

    residuals.append(np.sqrt(sum((A_reshaped.dot(u_reshaped) - B_reshaped) ** 2)))
    # calculating square error
    errors_2d = []
    denominator = []

    delta2n = delta ** -2
    for k in range(N):
        errors_2d.append([])
        for j in range(N):
            x = (a + k * delta)
            y = (a + j * delta)
            expected = g(x, y)

            if x ** 2 + y ** 2 < 0.9:
                errors_2d[-1].append(abs(abs(expected - u[k][j]) / expected))
                denominator.append(expected ** 2)
            else:
                errors_2d[-1].append(0)

    print(errors_2d)

    draw_plot(xs, ys, errors_2d)

    es.append(sum(np.sum(errors_2d)) / sum(denominator))
    # deltas.append(1 / N)
    deltas.append(N)

print(es)
# print(residuals)

# plt.yscale('log')
# plt.xscale('log')
# plt.plot(deltas, es)
# plt.show()
