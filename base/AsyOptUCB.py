from math import log, sqrt
from random import choice

'''
Asymptotically Optimal UCB Algorithm:
input: 
    bandit -- the bandit to play
    n -- number of total rounds
'''


def AsyOptUCB(bandit, n):
    k = bandit.K
    results = [0] * k
    ucb = [0] * k
    T = [0] * k

    # Choose each arm once
    for i in range(k):
        results[i] += bandit.pull(i)
        T[i] += 1
        f = 1 + (i + 1) * log(i + 1) ** 2
        ucb[i] = results[i] / T[i] + sqrt(2 * log(f) / T[i])

    for i in range(k + 1, n + 1):
        # choose action of max UCB
        options = [i for i, x in enumerate(ucb) if x == max(ucb)]
        opt = choice(options)

        # observe reward
        results[opt] += bandit.pull(opt)
        T[opt] += 1

        # update ucb
        f = 1 + i * log(i) ** 2
        ucb[opt] = results[opt] / T[opt] + sqrt(2 * log(f) / T[opt])
