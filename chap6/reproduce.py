from base.ETC import *
from base.GaussianBandit import GaussianBandit
import matplotlib.pyplot as plt

'''
Fig. 6.1 shows the expected regret of ETC when playing a Gaussian bandit 
with k = 2 and means µ1 = 0 and µ2 = −∆. The horizon is set to n = 1000 
and the suboptimality gap ∆ is varied between 0 and 1. 
Each data point is the average of 105 simulations, which makes the error 
bars invisible. The results show that the theoretical upper bound provided 
by Theorem 6.1 is quite close to the actual performance.
'''

if __name__ == '__main__':
    # parameters
    n = 1000  # horizon
    trials = 10 ** 4  # number of trials
    delta = [i / 100 for i in range(1, 101)]  # range of delta

    # results   
    actual_regret = []
    bounded_regret = []

    for d in delta:
        regret = []
        m = optimal_m(n, d)
        bounded_regret.append(regret_upper_bound(d, n))
        for t in range(trials):
            bandit = GaussianBandit([0, -d])
            ETC(bandit, n, m)
            regret.append(bandit.regret())
        actual_regret.append(sum(regret) / len(regret))

    fig, ax = plt.subplots()
    plt.plot(actual_regret, label='ETC')
    plt.plot(bounded_regret, label='Upper bound')
    legend = ax.legend()
    plt.show()
