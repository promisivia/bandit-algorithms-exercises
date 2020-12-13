from math import log, inf, sqrt
from random import choice

'''
Upper Confidence Bound Algorithm:
input: 
    bandit -- the bandit to play
    n -- number of total rounds
    delta -- parameter
'''


def UCB(bandit, n, delta):
    k = bandit.K
    results = [0] * k
    ucb = [inf] * k
    T = [0] * k

    # playing each arm a ďŹxed m number of times 
    for i in range(n):
        # choose action of max UCB
        options = [i for i, x in enumerate(ucb) if x == max(ucb)]
        opt = choice(options)

        # observe reward
        results[opt] += bandit.pull(opt)
        T[opt] += 1

        # update ucb
        ucb[opt] = results[opt] / T[opt] + sqrt(2 * log(1 / delta) / T[opt])
