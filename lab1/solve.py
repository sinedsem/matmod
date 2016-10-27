import numpy as np
import matplotlib.pyplot as plt

a = [[2.32, 0.12, 1.57], [1.69, 4.17, -0.33], [0.45, 1.82, 3.15]]
b = [2.01, 3.23, 0.94]

#print(np.linalg.solve(a, b))


def solve(a, b):
    result = []
    x = np.zeros(len(b))

    p = np.zeros((len(a), len(a)))

    for i in range(len(a)):
        p[i][i] = 1 / a[i][i]

    for k in range(100):
        x = x + np.dot(p, b) - np.dot(np.dot(p, a), x)
        result.append(sum((np.dot(a, x) - b) ** 2))
        if result[-1] < 10 ** -29:
            break

    plt.plot(range(1, len(result) + 1), result)
    plt.yscale('log')
    plt.show()

solve(a, b)
