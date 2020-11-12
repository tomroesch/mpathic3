import numpy as np

def mutualinfo(pxy, tol=1E-10):
    """
    Computes mutual information given a 2d probability distribution 
    """
    #with open('test_error','w') as f:
        #np.savetxt(f,raw_pxy)
    # Fix up probability distributions for entropy calculations
    px = fix_probs(pxy.sum(axis=1))
    py = fix_probs(pxy.sum(axis=0))

    # Compute mutual information and check for anomalies
    mi = entropy(px) + entropy(py) - entropy(pxy)

    # If MI is near zero, set to zero.
    if abs(mi) < tol:
        mi = 0.0

    # Do sanity check
    assert mi >= 0.0
    return mi


def entropy(raw_ps):
    """
    Compute entropy given a probability distribution
    """

    # Fix up probability distribution
    ps = fix_probs(raw_ps)

    # Get nonzero elements
    indices = ps>0.0

    # Compute entropy
    ent = -np.sum(ps[indices]*np.log2(ps[indices]))

    # Check for anomalies
    assert ent >= 0.0

    return ent


def fix_probs_2d(ps):
    """
    Make sure probability distribution is 2d and valid
    """
    ps /= ps.flatten().sum()
    return ps

def fix_probs(ps):
    """
    Make sure probability distribution is 1d and valid
    """
    ps /= ps.sum()
    return ps