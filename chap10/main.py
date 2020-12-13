from base.BernoulliBandit import BernoulliBandit
from base.KLUCB import *
from base.AsyOptUCB import AsyOptUCB

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # parameters
    n = 10000  # horizon
    trials = 10 ** 1  # number of trials
    mean = .5
    delta = [i / 100 for i in range(100)]  # range of delta

    asy_ucb_regret = []
    klucb_regret = []

    regret = []
    for d in delta:
        print("running with delta = ", d)

        # Asy Opt UCB
        regret.clear()
        for t in range(trials):
            bandit = BernoulliBandit([mean, mean + d])
            AsyOptUCB(bandit, n)
            regret.append(bandit.regret())
        asy_ucb_regret.append(sum(regret) / len(regret))
        print("asy_ucb_regret now is: ", asy_ucb_regret)

        # KLUCB
        regret.clear()
        for t in range(trials):
            bandit = BernoulliBandit([mean, mean + d])
            KLUCB(bandit, n)
            regret.append(bandit.regret())
        klucb_regret.append(sum(regret) / len(regret))
        print("klucb_regret now is: ", klucb_regret)

    fig, ax = plt.subplots()
    plt.xlim((0, 1))
    x = np.arange(0, 1, 0.01)
    plt.plot(x, asy_ucb_regret, label='Asy Opt UCB', color='#EDB120')
    plt.plot(x, klucb_regret, label='KLUCB', color='#0072BD')

    legend = ax.legend()
    plt.xlabel('delta')
    plt.ylabel('Expected regret')
    plt.savefig("regret_bound_with_μ={}_trials={}.png".format(mean, trials), dpi=400)
    plt.show()
