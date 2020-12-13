from math import log, sqrt

'''
Modified UCB Algorithm:
input: 
    bandit -- the bandit to play
    n -- number of total rounds
'''


def ModifiedUCB(bandit, n):
    results = [0] * 2
    T = [0] * 2

    results[0] += bandit.pull(0)
    T[0] += 1

    for t in range(2, n + 1):
        opt = 1
        # if greater than 0, play the first arm
        if float(results[0]) / T[0] + sqrt(2 * log(1 + t * log(t) ** 2) / T[0]) >= 0:
            opt = 0
        results[opt] += bandit.pull(opt)
        T[opt] += 1
