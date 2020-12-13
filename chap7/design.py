from base.UCB import UCB
from base.GaussianBandit import GaussianBandit
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    d = 1 / 10
    n = 1000
    trials = 10 ** 4
    interval = 1 / (100 * n)

    regret_c = []
    for delta in np.arange(interval, 1 / n, interval):
        res = []
        for t in range(trials):
            bandit = GaussianBandit([0, -d])
            UCB(bandit, n, delta)
            res.append(bandit.regret())
        avg = sum(res) / len(res)
        regret_c.append(avg)

    plt.xlim((0, 2 / n))
    x = np.arange(interval, 2 / n, interval)
    plt.plot(x, regret_c, label='ETC regret')

    plt.xlabel('delta')
    plt.ylabel('Expected regret')
    plt.legend()
    plt.savefig("design.png", dpi=400)
    plt.show()
