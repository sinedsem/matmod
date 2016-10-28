import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math


def get_B(a, b, N):
    result = np.zeros((N, N))

    delta = (b - a) / N
    delta2 = delta ** 2
    delta2n = delta ** -2

    xs = [a + delta * x for x in range(N)]
    ys = [a + delta * x for x in range(N)]

    for k in range(N):
        for j in range(N):
            if (k - N / 2) ** 2 + (j - N / 2) ** 2 < delta2n:
                # if True:
                result[k][j] = delta2

    # result[0] -= fa / (delta ** 2)
    # result[-1] -= fb / (delta ** 2)

    return delta, result, xs, ys


def make_A(N, delta):
    delta2 = delta ** 2
    delta2n = delta ** -2
    A = np.zeros((N, N, N, N))
    for k in range(N):
        for j in range(N):
            for k1 in range(N):
                for j1 in range(N):
                    if k == k1 and j == j1:
                        # A[k][k1][j][j1] = -4
                        A[k][j][k1][j1] = -4
                    elif abs(k - k1) == 1 and j == j1 and (k - N / 2) ** 2 + (j - N / 2) ** 2 < delta2n:
                        # elif abs(k - k1) == 1 and j == j1:
                        # A[k][k1][j][j1] = 1
                        A[k][j][k1][j1] = 1
                    elif abs(j - j1) == 1 and k == k1 and (k - N / 2) ** 2 + (j - N / 2) ** 2 < delta2n:
                        # elif abs(j - j1) == 1 and k == k1:
                        # A[k][k1][j][j1] = 1
                        A[k][j][k1][j1] = 1

    return A


a = -1
b = 1


# g = lambda x, y: (x ** 2 + y ** 2 - 1) / 4

def g(x, y):
    if x ** 2 + y ** 2 < 1:
        return (x ** 2 + y ** 2 - 1) / 4
    else:
        return 0


es = []
ns = []

n = 80


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


for N in range(n, n + 1, 1):
    delta, B, xs, ys = get_B(a, b, N)
    A = make_A(N, delta)

    A_reshaped = np.reshape(A, (N ** 2, N ** 2))
    B_reshaped = np.reshape(B, (N ** 2))

    # print(A_reshaped)
    # print(B_reshaped)

    u_reshaped = np.linalg.solve(A_reshaped, B_reshaped)
    # print(u_reshaped)
    u = np.reshape(u_reshaped, (N, N))

    # draw_plot(xs, ys, u)

    print(ys)

    z = [[g(x, y) for y in ys] for x in xs]
    # draw_plot(xs, ys, z)

    print(np.min(u))

    print(min([min(zz) for zz in z]))

    errors = []
    delta2n = delta ** -2
    for k in range(N):
        print()
        for j in range(N):
            if (k - N / 2) ** 2 + (j - N / 2) ** 2 < delta2n:
                x = (a + k * delta)
                y = (a + j * delta)
                expected = g(x, y)
                # print(expected, u[k][j])
                if expected == 0:
                    continue
                errors.append(abs(expected - u[k][j]) / expected)

    es.append(max(errors))
    ns.append(N)

print(es)

# plt.yscale('log')
# plt.plot(ns, es)
# plt.show()
