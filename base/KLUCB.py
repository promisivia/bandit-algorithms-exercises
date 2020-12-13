from math import log, inf
from random import choice

eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]


# 输入两个值，返回他们的kl距离
def klDistance(p, q):
    # print("p/mean:{}, q/μ:{}".format(p, q))

    # method 1: 通过加入一个极小的常量，使 [0, 1] 变为 [eps, 1 - eps]
    # p = min(max(p, eps), 1 - eps)
    # q = min(max(q, eps), 1 - eps)
    # return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))

    # method 2: 讨论所有的特殊情况
    if q != 0 and q != 1:
        if p == 0:  # q == 0, p in (0, 1)
            return log(1 / (1 - q))
        elif p == 1:  # q == 1, p in (0, 1)
            return log(1 / q)
        else:  # q in (0, 1), p in (0, 1)
            return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
    elif q == 0:
        if p == 0:  # q == 0, p == 0
            return 0
        else:
            return inf
    else:  # q == 1
        if p == 1:  # q == 1, p == 1
            return 0
        else:
            return inf


# 返回 f(t)
def f(t):
    return 1 + t * log(t) ** 2


# 返回和x距离小于d, 上限下限已知的最大值
def klucb(x, d, upper_bound, precision=1e-6, lower_bound=float('-inf'), max_iterations=50):
    value = max(x, lower_bound)
    u = upper_bound
    _count_iteration = 0
    while _count_iteration < max_iterations and u - value > precision:
        _count_iteration += 1
        m = (value + u) * 0.5
        if klDistance(x, m) > d:
            u = m
        else:
            value = m
    return (value + u) * 0.5


def KLUCB(bandit, n):
    k = bandit.K
    results = [0] * k
    ucb = [1] * k
    T = [0] * k

    # Choose each arm once
    for i in range(k):
        results[i] += bandit.pull(i)
        T[i] += 1
        d = log(f(i + 1))
        ucb[i] = klucb(results[i], d, upper_bound=1, lower_bound=0)

    for i in range(k + 1, n + 1):
        # choose action of max UCB
        options = [i for i, x in enumerate(ucb) if x == max(ucb)]
        opt = choice(options)

        # observe reward
        results[opt] += bandit.pull(opt)
        T[opt] += 1

        # update ucb
        mean = results[opt] / T[opt]
        d = log(f(i + 1)) / T[opt]
        ucb[opt] = klucb(mean, d, upper_bound=1, lower_bound=0)
