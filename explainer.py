import numpy as np
from sampler import Sampler
from kernel import Kernel
from surrogate import LocalSurrogate

class LimeExplainer:
    def __init__(self, predict_fn, x, baseline, n, feature_frac, sigma, lam):
        sampler = Sampler(x, n, baseline, feature_frac)
        
        y = np.array([predict_fn(p) for p in sampler.perturbations])
        
        kernel = Kernel(sampler.dist_sq_perturbations, sigma)
        
        surrogate = LocalSurrogate(sampler.masks, y, kernel.weights, lam)
        
        self.sampler = sampler
        self.kernel = kernel
        self.surrogate = surrogate
        
        self.beta = surrogate.beta