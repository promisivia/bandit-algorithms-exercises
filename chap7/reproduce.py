from base.ETC import *
from base.UCB import UCB
from base.GaussianBandit import GaussianBandit

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # parameters
    n = 1000  # horizon
    trials = 10 ** 4  # number of trials
    delta = [i / 100 for i in range(1, 101)]  # range of delta
    # results   
    etc_opt_regret = []
    etc_m25_regret = []
    etc_m50_regret = []
    etc_m75_regret = []
    etc_m100_regret = []
    ucb_regret = []

    for d in delta:
        print("running delta = ", d)
        # optimal m
        regret = []
        m = optimal_m(n, d)
        for t in range(trials):
            bandit = GaussianBandit([0, -d])
            ETC(bandit, n, m)
            regret.append(bandit.regret())
        etc_opt_regret.append(sum(regret) / len(regret))

        # m = 25
        regret.clear()
        for t in range(trials):
            bandit = GaussianBandit([0, -d])
            ETC(bandit, n, 25)
            regret.append(bandit.regret())
        etc_m25_regret.append(sum(regret) / len(regret))

        # m = 50
        regret.clear()
        for t in range(trials):
            bandit = GaussianBandit([0, -d])
            ETC(bandit, n, 50)
            regret.append(bandit.regret())
        etc_m50_regret.append(sum(regret) / len(regret))

        # m = 75
        regret.clear()
        for t in range(trials):
            bandit = GaussianBandit([0, -d])
            ETC(bandit, n, 75)
            regret.append(bandit.regret())
        etc_m75_regret.append(sum(regret) / len(regret))

        # m = 100
        regret.clear()
        for t in range(trials):
            bandit = GaussianBandit([0, -d])
            ETC(bandit, n, 100)
            regret.append(bandit.regret())
        etc_m100_regret.append(sum(regret) / len(regret))

        # ucb
        regret.clear()
        for t in range(trials):
            bandit = GaussianBandit([0, -d])
            UCB(bandit, n, 1 / n)
            regret.append(bandit.regret())
        ucb_regret.append(sum(regret) / len(regret))

    fig, ax = plt.subplots()
    plt.xlim((0, 1.01))
    x = np.arange(0.01, 1.01, 0.01)
    plt.plot(x, etc_opt_regret, label='ETC(optimal m)', color='grey')
    plt.plot(x, etc_m25_regret, label='ETC(m=25)', color='blue')
    plt.plot(x, etc_m50_regret, label='ETC(m=50)', color='green')
    plt.plot(x, etc_m75_regret, label='ETC(m=75)', color='yellow')
    plt.plot(x, etc_m100_regret, label='ETC(m=100)', color='red')
    plt.plot(x, ucb_regret, label='UCB', color='black')

    legend = ax.legend()
    plt.xlabel('?')
    plt.ylabel('Expected regret')
    plt.savefig("reproduce.png", dpi=400)
    plt.show()
