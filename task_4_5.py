import numpy as np
import pandas as pd
import scipy

from scipy import special, integrate
from scipy.misc import derivative
from matplotlib import pyplot as plt
import mpmath


def F(p):
    return (np.power(p, 1 / 4) - 2) / (np.power(p, 3) - 2)


def phi(p):
    # return p * F(p)
    return (np.power(p, 5 / 4) - 2 * p) / (np.power(p, 3) - 2)


def f_sum_1(t, eps=0.001):
    result = 0
    prev = -1
    k = 0

    while np.abs(result - prev) > eps:
        prev = result
        add1 = np.power(2, k) * np.power(t, (7 / 4) + 3 * k) / special.gamma((11 / 4) + 3 * k)
        add2 = np.power(2, k) * np.power(t, 2 + 3 * k) / special.gamma(3 + 3 * k)
        result = result + add1 - 2 * add2
        k = k + 1

    return result


def f_sum(t, eps=0.00001):
    result = 0
    prev = -1
    k = 0

    add1 = np.power(t, (7 / 4)) / special.gamma((11 / 4))
    add2 = - 2 * np.power(t, 2 + 3 * k) / special.gamma(3)

    while np.abs(result - prev) > eps:
        prev = result
        # add1 = np.power(2,k)*np.power(t, (7 / 4) + 3 * k) / special.gamma((11 / 4) + 3 * k)
        # add2 = np.power(2,k)*np.power(t, 2 + 3 * k) / special.gamma(3 + 3 * k)
        # result = result + add1 - 2 * add2
        # k = k + 1

        result = result + add1 + add2
        add1 = add1 * 2 * np.power(t, 3) / (11 / 4 + 3 * k) / (11 / 4 + 3 * k + 1) / (11 / 4 + 3 * k + 2)
        add2 = add2 * 2 * np.power(t, 3) / (3 + k * 3) / (3 + k * 3 + 1) / (3 + k * 3 + 2)
        k = k + 1

    return result


def em(m, x):
    res = np.exp(complex(0, 1) * 2 * x * np.pi / m)
    return res


def widder(t, n, m, r):
    W = 0
    for i in range(1, m + 1):
        add = (np.power(r * em(m, i), -n) / m * phi(n * (1 - r * em(m, i)) / t) / (1 - r * em(m, i)))
        W = W + add
        # add = em(m, -n * i) * F(n * (1 - r * em(m, i)) / r)
        # W = W + add * n / (m * t * np.power(r, n))
    W = W
    return W


def widder_cheb(t, n, r, k, n1=0.5):
    xs = [1 / 2 * ((1 / n1 - 1 / n) * np.cos(i * np.pi / (k - 1)) + 1 / n1 + 1 / n) for i in range(k)];
    ms = [1. / x for x in xs]
    d = [m / ms[0] for m in ms]
    num = [int(np.ceil(m + 1 / 2)) for m in ms]
    res = 0
    c = np.ones(k)
    for j in range(k):
        for i in range(k):
            if (i != j):
                c[j] = c[j] * d[j] / (d[j] - d[i])
    for j in range(k):
        res = res + (widder(t, num[j], 2 * num[j], r)) * c[j]
    return res


def widder_best(t, n, r, k):
    nums = range(n - k, n)
    d = [num / n for num in nums]
    res = 0
    c = np.ones((k))
    for j in range(k):
        for i in range(k):
            if (i != j):
                c[j] = c[j] * d[j] / (d[j] - d[i])
    for j in range(k):
        res = res + (widder(t, nums[j], 2 * nums[j], r)) * c[j]
    return res


def momentum(m, a, b, alpha=0.0001):
    # r = np.log(2) / (b-a)
    r = (b - a) / m
    # r = 1 / b
    A = np.ndarray((m, m))
    for k in range(m):
        for j in range(m):
            A[k, j] = integrate.quad(lambda x: np.power(x, k) * scipy.special.legendre(j)(2 * x - 1), 0, 1)[0]
            # A[k, j] = \
            # integrate.quad(lambda x: np.power(x, k) * scipy.special.legendre(j)(2 * (x - a) / (b - a) - 1), 0, 1)[0]
    equation_matrix = np.matmul(np.transpose(A), A) + alpha * np.eye(m)
    u = np.array([F(a + r * i) for i in range(m)]).transpose()
    equation_right = np.matmul(np.transpose(A), u)
    c = np.linalg.solve(equation_matrix, equation_right)

    return lambda t: momenum_result(c, m, r, t)


def P(j, x):
    if j == 0:
        return 1
    if j == 1:
        return x
    return (2 * (j - 1) + 1) / j * P(j - 1, x) - (j - 1) / j * P(j - 2, x)


def momenum_result(c, m, r, t):
    x = np.exp(-r * t)
    h = 0
    for i in range(m):
        h = h + c[i] * scipy.special.legendre(i)(2 * x - 1)
        # h = h + c[i] *scipy.special.legendre(i)(2 * (x - a) / (b - a) - 1)
    return r * np.exp(a * t) * h


a = 2
b = 2.5
r = 0.8
k = 3
n = 30

ts = np.linspace(a, b)
# analytycal_1 = [f_sum_1(t) for t in ts]
# plt.plot(ts, analytycal_1, label="аналитическое 1")
analytycal = np.array([f_sum(t) for t in ts])

plt.plot(ts, analytycal, label="аналитическое")

W = np.array([widder(t, n, 2 * n, r) for t in ts])
plt.plot(ts, W, label="Виддера")

W_c = np.array([widder_cheb(t, n * 2, r, k, n1=1 / (b + 0.5)) for t in ts])
plt.plot(ts, W_c, label="Виддера с Чебышёвскими номерами")

W_b = np.array([widder_best(t, n, r, k) for t in ts])
plt.plot(ts, W_b, label="Виддера с наилучшими номерами")

# momentum_1  =momentum(10,0)
# mom = np.array([momentum_1(t) for t in ts])
# plt.plot(ts, mom, label="Метод моментов")

momentum_reg = momentum(30, a, b, alpha=0.0000000000001)
mom_reg = np.array([momentum_reg(t) for t in ts])
plt.plot(ts, mom_reg, label="Метод моментов с регуляризацией")

# L = [mpmath.invertlaplace(F, t) for t in ts]
# plt.plot(ts, L, label="reshenie")
plt.legend()
plt.show()

ns = [40, 50, 60, 70, 80]
# ns = [40]
result = np.ndarray((len(ns), 5))
for i, n in enumerate(ns):
    W = np.array([widder(t, n, 2 * n, r) for t in ts])
    W_c = np.array([widder_cheb(t, n * 2, r, k, n1=1 / (b + 0.5)) for t in ts])
    W_b = np.array([widder_best(t, n, r, k) for t in ts])
    momentum_reg = momentum(n, a, b, alpha=0.0000000000001)
    mom_reg = np.array([momentum_reg(t) for t in ts])

    w_err = np.max(np.abs(W - analytycal))
    wb_err = np.max(np.abs(W_b - analytycal))
    wc_err = np.max(np.abs(W_c - analytycal))
    moment_err = np.max(np.abs(mom_reg - analytycal))
    result[i, 0] = n
    result[i, 1] = w_err
    result[i, 2] = wc_err
    result[i, 3] = wb_err
    result[i, 4] = moment_err

df = pd.DataFrame(result)
df.columns = ['N', "Без ускорения", "Чебышёвские параметры", "Наилучшие параметры", "Метод моментов"]
pd.set_option('display.max_columns', None)
print(df)
