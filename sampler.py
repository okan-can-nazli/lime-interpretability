import numpy as np

class Sampler:
    
    def __init__(self, x, n, baseline, feature_frac = 0.5):
        """_
        Args:
            x (input data matrix)
            n (sample number)
            baseline: masked-off value of a feature
            feature_frac : probability feature gets mask or not.(stay same value on 1 changes on 0)
        """
        self.x = x
        self.n = n
        self.baseline = baseline
        self.feature_frac = feature_frac
        
        self.perturbations = []
        self.dist_sq_perturbations = []
        self.masks = [] 
            
        for i in range(n):
            
            mask_i = np.random.binomial(1, feature_frac, size=x.shape) # binary matrix (value 1 depends on feature_frac)
            neighbour = x * mask_i + (1 - mask_i) * baseline # apply baseline to masked vlaues
            distance_sq_i = np.sum((mask_i - np.ones_like(mask_i))**2)


            self.perturbations.append(neighbour)
            self.dist_sq_perturbations.append(distance_sq_i) # save perturbation dist
            self.masks.append(mask_i)
            
        
