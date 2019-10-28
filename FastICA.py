import numpy as np

def whiten(X, epsilon=None, method='zca'):
    """Transforms a vector of random variables into a set of new variables whose
    covariance is the identity matrix.

    Keyword arguments:
    X -- the random variables vector that will be transformed
    epsilon -- small factor to avoid division by zero (default machine epsilon)
    method -- method for the transformation (zca, pca or cholesky)
    """
    if not epsilon:
        epsilon = np.finfo(float).eps
        
    if method not in ('zca', 'pca', 'cholesky'):
        raise ValueError("Method should be either zca, pca or cholesky")
        
    cov = X.dot(X.T) / float(X.shape[1])
    W = None
    
    U, S, V = np.linalg.svd(cov)
    D = np.diag(1. / S + epsilon)
    
    if method == 'zca':
        W = U.dot(np.sqrt(D)).dot(U.T)
    elif method == 'pca':
        W = np.sqrt(D).dot(U.T)
    elif method == 'cholesky':
        L = np.linalg.cholesky(U.dot(D.dot(U.T)))
        W = L.T

    return W.dot(X)

def center(X, axis=1):
    """Centers a vector of random variables along an axis subtracting its mean
    value.

    Keyword arguments:
    X -- the random variable vector that will be centered
    axis -- axis in which the centering will occur (default 0)
    """
    return X - np.mean(X, axis=axis).reshape(-1, 1)

def kurtosis(X):
    def _moment(X, k):
        """Returns the kth moment of a random variable.
    
        Keyword arguments:
        X -- the random variable vector of which the kth moment will be extracted
        k -- the moment order
        """
        m = np.mean(X)
        return np.sum(np.power(X - m, k)) / len(X)

    return _moment(X, 4)/np.square(_moment(X, 2)) - 3

def g1(u, a1=1):
    """ G = (1/a1) * log(cosh(a1*u))
    """
    if a1 < 1 or a1 > 2:
        raise ValueError("a1 must be in [1, 2]")

    return (np.tanh(a1*u), a1 * (np.square(np.tanh(u))).mean(axis=-1))

def g2(u):
    """ G = -exp(-u**2 / 2)
    """
    exp = np.exp(-np.square(u)/2)
    return ((u * exp), ((1 - np.square(u))*exp).mean(axis=-1))

def g3(u):
    """ G = (1/4) * u**4
    """
    return (np.power(u, 3), 3 * np.power(u, 2).mean(axis=-1))

def g4(u):
    """ G = -cos(u)
    """
    return (np.sin(u), np.cos(u))

def normalizeAudio(Y, maximum=1):
    if maximum > 1 or maximum < 0:
        raise ValueError("Maximum should be in [0, 1]")
    factor = (Y.max(axis=1) - Y.min(axis=1))
    factor = factor.reshape(-1, 1) / (2 * maximum)
    return Y/factor

def fastICA(X, tol=1e-5, maxIters=10000, seed=None, fun=None):
    m, n = X.shape
    
    if seed: np.random.seed(seed)
    if fun == None: fun = g1
    
    W = np.random.rand(m, m)
    
    X_w = whiten(center(np.array(X)))
    
    iterations = 0
    for row in range(m):
        w = W[row, :].copy()
        w /= np.sqrt(np.sum(np.square(w)))
        
        for _ in range(maxIters):
            iterations += 1
            # Calculate g and dg
            g, dg = fun(w.T.dot(X_w))
    
            # Update weights
            wNew = (X_w * g).mean(axis=1) - (dg.mean() * w)
            
            # Decorrelate weights
            wNew -= wNew.dot(W[:row].T).dot(W[:row])
            
            # Normalize weights
            wNew /= np.sqrt(np.sum(np.square(wNew)))
            
            # Calculate tolerance condition
            error = np.abs((wNew * w).sum()) - 1
            
            w = wNew
            
            if np.abs(error) < tol:
                break
        
        W[row, :] = w
    print(iterations)
    return W.dot(X_w), W
