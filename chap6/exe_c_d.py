from base.ETC import *
from base.GaussianBandit import GaussianBandit
import numpy as np
import matplotlib.pyplot as plt
import statistics

'''
Fix delta = 1/10 and n =2000 
plot the expected regret as a function of m 
'''

if __name__ == '__main__':
    trials = 10 ** 5
    interval = 5
    # Part c
    d = 1 / 10
    n = 2000
    regretc = []
    error = []
    for m in range(10, 401, interval):
        res = []
        for t in range(trials):
            bandit = GaussianBandit([0, -d])
            ETC(bandit, n, m)
            res.append(bandit.regret())
        avg = sum(res) / len(res)
        regretc.append(avg)
        err = statistics.variance(res) ** .5
        error.append(err)
        print("m:", m, " avg:", avg, " error:", err)

    plt.xlim((0, 420))
    plt.xticks(np.arange(0, 420, 100))
    x = np.arange(10, 401, interval)
    plt.plot(x, regretc, label='ETC regret')

    # Part d
    plt.plot(x, error, label='ETC regret standard deviation')
    plt.xlabel('m')
    plt.legend()
    plt.show()
