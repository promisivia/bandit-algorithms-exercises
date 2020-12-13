from base.AsyOptUCB import AsyOptUCB
from base.UCB import UCB
from chap8.ModifiedUCB import ModifiedUCB
from chap8.OneArmBandit import OneArmBandit

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # parameters
    n = 1000  # horizon
    trials = 10 ** 4  # number of trials
    mean = [i / 100 for i in range(-100, 101)]  # range of mean of the first arm

    ucb_regret = []
    ucb_2_regret = []
    opt_ucb_regret = []
    modified_ucb_regret = []

    for m in mean:
        print("running with mean = ", m)
        # ucb
        regret = []
        for t in range(trials):
            bandit = OneArmBandit(m)
            UCB(bandit, n, 1 / n)
            regret.append(bandit.regret())
        ans = sum(regret) / len(regret)
        ucb_regret.append(ans)
        print("ucb with regret = ", ans)

        # ucb with delta = 1/n^2
        regret.clear()
        for t in range(trials):
            bandit = OneArmBandit(m)
            UCB(bandit, n, 1 / n ** 2)
            regret.append(bandit.regret())
        ans = sum(regret) / len(regret)
        ucb_2_regret.append(ans)
        print("ucb with regret = ", ans)

        # asy opt ucb
        regret.clear()
        for t in range(trials):
            bandit = OneArmBandit(m)
            AsyOptUCB(bandit, n)
            regret.append(bandit.regret())
        ans = sum(regret) / len(regret)
        opt_ucb_regret.append(ans)
        print("ucb with regret = ", ans)

        # modified ucb
        regret.clear()
        for t in range(trials):
            bandit = OneArmBandit(m)
            ModifiedUCB(bandit, n)
            regret.append(bandit.regret())
        ans = sum(regret) / len(regret)
        modified_ucb_regret.append(ans)
        print("modified ucb with regret = ", ans)

    fig, ax = plt.subplots()
    plt.xlim((-1, 1.01))
    x = np.arange(-1, 1.01, 0.01)
    plt.plot(x, ucb_regret, label='UCB with delta 1/n', color='red')
    plt.plot(x, ucb_2_regret, label='UCB with delta 1/n^2', color='yellow')
    plt.plot(x, opt_ucb_regret, label='Asymptotically Optimal UCB', color='green')
    plt.plot(x, modified_ucb_regret, label='Modified UCB', color='blue')

    legend = ax.legend()
    plt.xlabel('μ1')
    plt.ylabel('Expected regret')
    plt.savefig("result/main.png", dpi=400)
    plt.show()
