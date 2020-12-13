from math import log
from random import choice


# the upper bound of regard
def optimal_m(n, delta):
    m = int(4 / (delta ** 2) * log(n * (delta ** 2) / 4) + 1)
    return max(1, m)


# the upper bound of regard
def regret_upper_bound(delta, n):
    max1 = n * delta
    max2 = delta + 4 / delta * (1 + max([0, log(n * (delta ** 2) / 4)]))
    return min(max1, max2)


'''
Explore-then-commit Algorithm:
input: 
    bandit -- the bandit to play
    n -- number of total rounds
    m -- number of explore rounds
'''


def ETC(bandit, n, m):
    k = bandit.K
    results = [0] * k

    # playing each arm a fixed m number of times 
    for i in range(k):
        for j in range(m):
            results[i] += bandit.pull(i)

    # randomly choice the arm with the best reward 
    options = [i for i, x in enumerate(results) if x == max(results)]
    opt = choice(options)

    # playing the chosen arm the remaining round
    for i in range(m * k, n):
        bandit.pull(opt)
