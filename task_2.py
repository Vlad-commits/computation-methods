import numpy as np
import pandas as pd

nmax = 100 + 1


def my_a(n):
    A = np.zeros((n, n), complex)
    for i in range(n):
        for j in range(n):
            a = np.longdouble(2 * (i + 1))
            b = np.power(np.longdouble(n - j), -2)
            A[i, j] = np.power(a, b)
    return A


def matrix_square_method1(A):
    w, v = np.linalg.eig(A)
    B = np.matmul(np.matmul(v, np.diag(np.sqrt(w))), np.linalg.inv(v))
    return B


result = np.zeros((nmax - 1, 4), np.double)
for n in range(1, nmax):
    A = my_a(n)
    B = matrix_square_method1(A)
    condA = np.linalg.cond(A)
    condB = np.linalg.cond(B)
    err = np.linalg.norm(np.matmul(B, B) - A)
    result[n - 1, 0] = n
    result[n - 1, 1] = condA
    result[n - 1, 2] = condB
    result[n - 1, 3] = err

cnames = ['N', 'cond(A)', 'cond(B)', '||B^2-A||'];
df = pd.DataFrame(result)
df.columns = cnames
print(df)
