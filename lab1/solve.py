import numpy as np
import matplotlib.pyplot as plt

a = [[5, 2, 3], [2, 6, 1], [3, 1, 7]]
true_x = [1, 2, 3]
b = np.dot(a, true_x)


def find_magic_number(a):
    p = np.zeros((len(a), len(a)))
    for i in range(len(a)):
        p[i][i] = 1 / a[i][i]
    return np.linalg.norm(np.eye(3, 3) - np.dot(p, a), ord=2)


def solve(a, b):
    residual = []
    error = []
    x = np.zeros(len(b))

    p = np.zeros((len(a), len(a)))

    for i in range(len(a)):
        p[i][i] = 1 / a[i][i]

    for k in range(100):
        f_x = np.dot(p, b) - np.dot(np.dot(p, a), x)
        x = x + f_x
        residual.append(np.sqrt(sum((np.dot(a, x) - b) ** 2)))
        error.append(np.sqrt(sum((x - true_x) ** 2)))
        if residual[-1] < 10 ** -29:
            break

    plt.plot(range(1, len(residual) + 1), residual)
    plt.plot(range(1, len(error) + 1), error)
    plt.yscale('log')
    plt.show()


solve(a, b)
# print(find_magic_number(a))
