import numpy as np
import pandas as pd

sqrt2 = np.sqrt(2)
vals =[1.4, 17 / 12, np.sqrt(2)+0.00001]


def f1(sqrt2):
    return np.power((sqrt2 - 1) / (sqrt2 + 1), 3)


def f2(sqrt2):
    return np.power((sqrt2 - 1), 6)


def f3(sqrt2):
    return np.power((3 - 2 * sqrt2), 3)


def f4(sqrt2):
    return 99 - 70 * sqrt2


def f5(sqrt2):
    return 1 / np.power((sqrt2 + 1), 6)


def f6(sqrt2):
    return 1 / (99 + 70 * sqrt2)

array = np.ndarray((0,6))
for sqrt2 in vals:
    array = np.row_stack ((array,np.array([f1(sqrt2),
    f2(sqrt2),
    f3(sqrt2),
    f4(sqrt2),
    f5(sqrt2),
    f6(sqrt2)])))

df = pd.DataFrame(array)
df.columns =['f1','f2','f3','f4','f5','f6']
df.index = ['1.4', '17 / 12', 'np.sqrt(2)']
print(df)