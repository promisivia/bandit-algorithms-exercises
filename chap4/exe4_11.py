import matplotlib.pyplot as plt

from chap4.exe4_7 import BernoulliBandit
from chap4.exe4_8 import FollowTheLeader

if __name__ == '__main__':
    results = []
    for i in range(1000):
        bandit = BernoulliBandit([0.5, 0.6])
        FollowTheLeader(bandit, 100)
        results.append(bandit.regret())
    plt.hist(results, bins=100)
    plt.show()