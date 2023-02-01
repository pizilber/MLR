import numpy as np
from itertools import permutations

# calculates the error of an estimate beta_hat, invariant to the order of components
# can handle also overparameterized beta_hat
def errfun_knownBeta(beta_hat, beta):
    K = beta.shape[1]

    beta_hat_new = np.zeros((beta.shape[0], max(beta_hat.shape[1], K)))
    beta_hat_new[:, :K] = beta_hat
    beta_hat = beta_hat_new
    
    ps = set(permutations(range(K)))
    err = np.Inf
    for p in ps:
        # beta_temp = the first K components of current permutation
        beta_temp = beta_hat[:, p]
        beta_temp = beta_temp[:, :K]
        K_errors = np.linalg.norm(beta_temp - beta, axis=1)
        curr_err = np.mean(K_errors)
        if curr_err < err:
            err = curr_err

    return err


# return intersection of two label vectors
def suppfun(c_hat, c, K):
    c_hat = c_hat.reshape(c.shape)
    if K < 6: # otherwise it's too computationally expensive
        perm = match_permutation_by_intersect(c_hat, c, K)
        c_hat[c_hat > 0] = perm[c_hat[c_hat > 0]]  # don't rearrange outliers
        return np.count_nonzero(c_hat == c) / len(c)
    else:
        return -1

def match_permutation_by_intersect(c_hat, c, K):
# find permutation over 1:K such that c_hat best matches c

    # ignore outliers
    c = c[c_hat >= 0]
    c_hat = c_hat[c_hat >= 0]

    # choose best permutation
    ps = set(permutations(range(K)))
    best_intersect = -1
    for p in ps:
        p = np.array(p)
        curr_intersect = np.count_nonzero(p[c_hat] == c)
        if curr_intersect > best_intersect:
            best_intersect = curr_intersect
            best_perm = p

    return best_perm