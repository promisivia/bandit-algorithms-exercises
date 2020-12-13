import numpy as np
import matplotlib.pyplot as plt


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


lbda = [x / 10 for x in range(1, 10)]
subgraph = [331, 332, 333, 334, 335, 336, 337, 338, 339]
size = 30

for t in range(9):
    p = lbda[t] / size
    results = []
    trials = 100000
    for i in range(trials):
        x = sum(np.random.binomial(size=size, n=1, p=p))
        results.append(x)

    # subgraph
    plt.subplot(subgraph[t])

    x = np.arange(-3, 3, 0.05)
    y = normfun(x, size * p, size * p * (1 - p))
    plt.plot(x, y)

    plt.xlim((-3, 3))
    plt.xticks(np.arange(-3, 3, 1))
    counts, bins = np.histogram(results)
    plt.hist(bins[:-1], bins, weights=counts / trials)
    plt.title('lambda=' + str(lbda[t]))
    plt.plot()

plt.show()
