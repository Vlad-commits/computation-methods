import numpy as np
import numpy.linalg
import pandas as pd
import scipy.integrate as integrate


def solve(a, b, K, U, n, aplha):
    S, step = create_net(a, b, n)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = step * K(S[i], S[j])

    C_transpoed = np.transpose(C)
    equation_matrix = np.matmul(C_transpoed, C) + aplha * np.eye(n)
    equation_right = np.matmul(C_transpoed, U)

    Z = numpy.linalg.solve(equation_matrix, equation_right)
    return Z


def create_net(a, b, n):
    step = (b - a) / n
    S = [a + step / 2 + i * step for i in range(n)]
    return S, step


def create_u(a, b, K, z, n):
    step = (b - a) / n
    xs = [a + step / 2 + i * step for i in range(n)]

    return [integrate.quad(lambda s: K(x, s) * z(s), a, b)[0] for x in xs]


a = 0
b = 1
K = lambda x, s: 1 / (3 + x * s)
alphas = [0,0.001,0.00001,0.0000000001]




def printres(z_exact):
    ns = [10, 20, 50]
    arr = np.empty((len(alphas), len(ns) + 1))
    for i, alpha in enumerate(alphas):
        arr[i, 0] = alpha
        for j, n in enumerate(ns):
            U = create_u(a, b, K, z_exact, n)
            Z = solve(a, b, K, U, n, alpha)
            norm_err = np.linalg.norm(Z - [z_exact(s) for s in create_net(a, b, n)[0]])
            arr[i, j + 1] = norm_err
    df = pd.DataFrame(arr, columns=["alpha", "n=10", "n=20", "n=50"])
    print(df)


printres(lambda x: 1)

printres(lambda x: x*x +x)



