import numpy as np

# weighted least squares
def weighted_ls(X, y, w=[]):
    if w == []:
        w = np.ones(len(y),)
    ws = np.sqrt(w)
    WX = ws[:, np.newaxis] * X
    if len(y.shape) > 1: # y is a matrix
        wy = ws[:, np.newaxis] * y
    else:
        wy = ws * y
    beta = np.linalg.inv(WX.T @ WX) @ (WX.T @ wy)
    return beta

def detect_outliers(res, corrupt_frac):
# input: res of size (n, K) and corrupt_frac between 0 and 1
# estimates outliers and returns an array with 1 for outlier and 0 for inlier

    n = res.shape[0]
    outlier_indicator = np.zeros((n,), dtype=np.bool8)
    M = np.min(res, axis=1) # take minimal res component per sample
    if corrupt_frac > 0: # mark outliers
        outlier_supp = np.argpartition(M, -round(corrupt_frac*n))[-round(corrupt_frac*n):]
        outlier_indicator[outlier_supp] = 1
   
    return outlier_indicator

def cluster_by_beta(beta, X, y, corrupt_frac):
# cluster samples into components
# if robustness > 0, mark outliers with c_hat < 0
    K = beta.shape[1]
    res = np.zeros((len(y), K))
    for k in range(K):
        res[:, k] = abs(X @ beta[:, k] - y)
    outlier_indicator = detect_outliers(res, corrupt_frac)
    I = np.argmin(res, axis=1)
    I[outlier_indicator.astype(np.int8)] = -1
    return I

def OLS(X, y, K, c):
    # perform OLS to each component individually according to c
    beta_hat = np.zeros((X.shape[1], K))
    for k in range(K):
        beta_hat[:,k] = weighted_ls(X[c==k,:], y[c==k])
    return beta_hat
