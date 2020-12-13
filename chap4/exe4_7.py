"""
ex 4.7 "Bernoulli bandit"
"""
import random


class BernoulliBandit:
    # accepts a list of K >= 2 floats, each lying in [0,1] 
    def __init__(self, means):
        assert len(means) >= 2, 'Requires at least 2 arms'
        self.means = means
        self.actions = []
        self.rewards = []

    # Function should return the number of arms 
    @property
    def K(self):
        return len(self.means)

    # Accepts a parameter 0 <= a <= K-1 and returns the 
    # realisation of random variable X with P(X = 1) being 
    # the mean of the (a+1)th arm. 
    def pull(self, a):
        assert 0 <= a <= self.K - 1, 'a should belongs to [0,K-1]'
        self.actions.append(a)
        r = random.random()
        if r < self.means[a]:
            self.rewards.append(1)
            return 1
        else:
            self.rewards.append(0)
            return 0

    # Returns the regret incurred so far. 
    def regret(self):
        opt = max(self.means) * len(self.actions)
        return opt - sum([self.means[a] for a in self.actions])
