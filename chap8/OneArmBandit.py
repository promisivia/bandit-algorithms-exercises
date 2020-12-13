from numpy.random import normal


class OneArmBandit:
    # Accepts a mean value
    def __init__(self, mean):
        self.means = [mean, 0]
        self.actions = []
        self.rewards = []
    
    # Function should return the number of arms 
    @property
    def K(self): 
        return len(self.means)

    # Accepts a parameter 0 <= a <= K-1 and returns the
    # realization of random variable X with P(X=1) being
    # the mean of the (a+1)th arm
    def pull(self, a):
        assert 0 <= a <= self.K - 1, "a should belongs to [0,K-1]"
        self.actions.append(a)
        result = normal(loc=self.means[a])
        self.rewards.append(result)
        return result

    def random_regret(self):
        opt = len(self.actions)*max(self.means)
        random_regret = opt - sum(self.rewards)
        return random_regret

    # Returns the regret incurred so far. 
    def regret(self):
        opt = len(self.actions)*max(self.means)
        regret = opt - sum([self.means[a] for a in self.actions])
        return regret
