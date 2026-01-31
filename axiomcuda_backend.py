"""
Stub backend for testing - replaces compiled axiomcuda_backend.

This allows the Python API to work without building C++ extensions.
Uses numpy and scipy for computations.
"""

import numpy as np

# Core functions
def cuda_available():
    return False

def get_device_count():
    return 0

# Math functions
def mvgammaln(x, d):
    """Multivariate gammaln using scipy."""
    from scipy.special import gammaln
    x = np.asarray(x)
    result = np.sum(gammaln(x[..., None] - np.arange(d) / 2.0), axis=-1)
    result += d * (d - 1) / 4.0 * np.log(np.pi)
    return result

def mvdigamma(x, d):
    """Multivariate digamma using scipy."""
    from scipy.special import digamma
    x = np.asarray(x)
    return np.sum(digamma(x[..., None] - np.arange(d) / 2.0), axis=-1)

def inv_and_logdet(matrix, return_inverse=True, return_logdet=True):
    """Matrix inverse and logdet using scipy."""
    from scipy import linalg
    matrix = np.asarray(matrix)
    
    # Add small regularization for numerical stability
    if matrix.ndim == 2:
        matrix = matrix + np.eye(matrix.shape[0]) * 1e-6
        sign, logdet = np.linalg.slogdet(matrix)
        inv = np.linalg.inv(matrix) if return_inverse else None
    else:
        # Batch case
        batch = matrix.shape[:-2]
        mat = matrix.reshape(-1, matrix.shape[-2], matrix.shape[-1])
        invs = []
        dets = []
        for m in mat:
            m_reg = m + np.eye(m.shape[0]) * 1e-6
            if return_logdet:
                sign, ld = np.linalg.slogdet(m_reg)
                dets.append(ld)
            if return_inverse:
                invs.append(np.linalg.inv(m_reg))
        logdet = np.array(dets).reshape(batch + (1, 1)) if return_logdet else None
        inv = np.array(invs).reshape(batch + matrix.shape[-2:]) if return_inverse else None
    
    if return_inverse and return_logdet:
        return inv, logdet
    elif return_inverse:
        return inv
    else:
        return logdet

def batch_dot(x, y):
    """Batched matrix multiply."""
    x = np.asarray(x)
    y = np.asarray(y)
    return np.matmul(x, y)

# Random functions  
def split_key(key):
    """Key splitting."""
    import hashlib
    key_bytes = str(key).encode()
    hash1 = hashlib.md5(key_bytes + b'1').hexdigest()[:16]
    hash2 = hashlib.md5(key_bytes + b'2').hexdigest()[:16]
    return int(hash1, 16), int(hash2, 16)

def normal_batch(key, shape):
    """Normal random."""
    np.random.seed(key % (2**32))
    return np.random.randn(*shape).astype(np.float64)

def uniform_batch(key, shape, minval=0.0, maxval=1.0):
    """Uniform random."""
    np.random.seed(key % (2**32))
    return np.random.uniform(minval, maxval, shape).astype(np.float64)

def categorical(key, probs):
    """Categorical sampling."""
    np.random.seed(key % (2**32))
    probs = np.asarray(probs)
    return np.random.choice(len(probs), p=probs/probs.sum())

def randint(key, minval, maxval, shape):
    """Random integers."""
    np.random.seed(key % (2**32))
    return np.random.randint(minval, maxval, shape)

# NN functions
def softmax(x, axis=-1):
    """Softmax using scipy."""
    from scipy.special import softmax as scipy_softmax
    return scipy_softmax(x, axis=axis)

def one_hot(indices, num_classes):
    """One-hot encoding."""
    indices = np.asarray(indices)
    return np.eye(num_classes)[indices]

# Model stubs
class SMM:
    def __init__(self, *args, **kwargs):
        pass

class RMM:
    def __init__(self, *args, **kwargs):
        pass

class TMM:
    def __init__(self, *args, **kwargs):
        pass

class IMM:
    def __init__(self, *args, **kwargs):
        pass

# Create module-like interface
models = type('models', (), {
    'SMM': SMM,
    'RMM': RMM,
    'TMM': TMM,
    'IMM': IMM,
})()
