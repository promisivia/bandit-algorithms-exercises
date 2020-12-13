"""
Using a horizon of n = 100, run 1000 simulations of your implementation of
Follow-the-Leader on the Bernoulli bandit above and record the regret
"""
import matplotlib.pyplot as plt
import statistics as st

from base.BernoulliBandit import BernoulliBandit
from chap4.exe4_8 import FollowTheLeader

if __name__ == '__main__':
    trials = [i * 100 for i in range(1, 11)]
    averaged_results = []
    error = []
    for k in trials:
        results = []
        for i in range(1000):
            bandit = BernoulliBandit([0.5, 0.6])
            FollowTheLeader(bandit, k)
            results.append(bandit.regret())
        averaged_results.append(st.mean(results))
        error.append(st.stdev(results) / len(results) ** 0.5)

    plt.errorbar(x=trials, y=averaged_results, yerr=error, capsize=3)
    plt.show()
