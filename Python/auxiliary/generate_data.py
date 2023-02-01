import numpy as np
from auxiliary.metrics import errfun_knownBeta, suppfun

def generate_data(cnfg):

    # set configurations
    n = cnfg['n']
    d = cnfg['d']
    K = cnfg['K']
    overparam = cnfg['overparam']
    distrib = cnfg['distrib']
    noise_level = cnfg['noise_level']
    corrupt_frac = cnfg['corrupt_frac']

    # produce labels
    c = np.random.choice(K, n, replace=True, p=distrib)

    # produce matrix X
    X = np.random.normal(size=(n,d))
    
    # produce K beta coefficients vectors
    beta = np.random.normal(size=(d,K))

    # produce (noisy) mixed observations
    Y = np.zeros((n,K))
    for k in range(K):
        Y[:,k] = X @ beta[:,k] + noise_level * np.random.normal(size=n)
    y = np.zeros((n,))
    for i in range(n):
        y[i] = Y[i, c[i]]

    # add corruptions (outliers)
    y_corrupted, c = corrupt(y, c, corrupt_frac)

    # evaluation metrics
    metrics = dict()
    metrics['errfun'] = lambda beta_hat: errfun_knownBeta(beta_hat, beta)
    metrics['suppfun'] = lambda c_hat: suppfun(c_hat, c, K + max(0, overparam))

    return X, y, beta, c, metrics


def corrupt(y, c, frac):
# add corruptions (outliers)
    n = len(y)
    corrupt_size = round(frac*n)
    currupt_support = np.random.choice(n, corrupt_size, False)
    y[currupt_support] = np.random.normal(scale=np.std(y), size=corrupt_size)
    c[currupt_support] = -1
    return y, c
