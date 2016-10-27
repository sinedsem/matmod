import numpy as np

N = 3
A = np.zeros((N, N, N, N))
A[0][0][2][2] = 1
# A[0][0][1][1] = 2
A_reshaped = np.reshape(A, (N ** 2, N ** 2))

B = np.zeros((N, N))
B[0][1] = 1
# A[0][0][1][1] = 2
B_reshaped = np.reshape(B, (N ** 2))


print(A_reshaped)
print(B_reshaped)