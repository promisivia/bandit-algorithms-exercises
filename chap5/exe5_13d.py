from math import pow
import numpy as np
import matplotlib.pyplot as plt


def calculate_n(n, mode, d):
    event = 0
    delta = 0.1
    p = 0.1
    for _iter in range(10000):
        if mode == 'binomial':
            x = np.random.binomial(p=p, n=1, size=n)
        elif mode == 'normal':
            x = np.random.normal(p, scale=p * (1 - p) / n, size=n)
        else:
            assert 0
        if (sum(x) / n) >= p + delta:
            event += 1
    return (event / n) < d


def test(mode, _delt, iterations=1000000):
    for _iter in range(1, iterations):
        if calculate_n(i, mode, _delt):
            return i


x_list = []
y_list = []
for i in range(10, 20):
    delt = 1 / pow(10, i)
    x_list.append(i)
    n1 = test('binomial', delt)
    n2 = test('normal', delt)
    y_list.append(n1 / n2)

plt.title("5-13d")
plt.xlabel("10^x")
plt.ylabel("n1/n2")
plt.plot(x_list, y_list)
plt.show()
