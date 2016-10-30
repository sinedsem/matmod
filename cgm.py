import numpy as np
from scipy.sparse import csr_matrix


def solve_cgm(A, b, eps=10 ** -14):
    x = list(b)
    r = [b - np.dot(A, x[0])]
    z = [r[0]]

    for k in range(1, len(b) + 1):
        alpha = sum(np.multiply(r[k - 1], r[k - 1])) / sum(np.multiply(np.dot(A, z[k - 1]), z[k - 1]))
        r.append(r[k - 1] - np.dot(np.dot(alpha, A), z[k - 1]))
        x.append(x[k - 1] + np.dot(alpha, z[k - 1]))
        beta = sum(np.multiply(r[k], r[k])) / sum(np.multiply(r[k - 1], r[k - 1]))
        z.append(r[k] + np.dot(beta, z[k - 1]))

        residual = np.sqrt(sum((np.dot(A, x[-1]) - b) ** 2))

        if residual < eps:
            break

    return x[-1]


def quick_solve_cgm(A, b, eps=10 ** -14):
    prev_x = list(b)
    prev_r = list(b - A.dot(prev_x))
    prev_z = list(prev_r)

    A = csr_matrix(A)

    for i in range(len(b)):
        if i % 10 == 0:
            print(i, "/", len(b), sep="")

        alpha = sum(np.multiply(prev_r, prev_r)) / sum(np.multiply(A.dot(prev_z), prev_z))
        r = prev_r - A.multiply(alpha).dot(prev_z)
        x = prev_x + np.dot(alpha, prev_z)
        beta = sum(np.multiply(r, r)) / sum(np.multiply(prev_r, prev_r))
        z = r + np.multiply(beta, prev_z)

        residual = np.sqrt(sum((A.dot(prev_x) - b) ** 2))
        if residual < eps:
            break

        prev_x = x
        prev_r = r
        prev_z = z

    return x
