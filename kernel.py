import numpy as np

class Kernel:
    def __init__(self, dist_sq_perturbations, sigma):
        """_
        Args:
            dist_sq_perturbation 
            sigma : softness of distance weight decider"kinda"
        """
        
        self.dist_sq_perturbations = dist_sq_perturbations
        self.sigma = sigma
        
        
        self.weights = np.exp(-np.array(dist_sq_perturbations) / sigma**2) # (0, 1]
        
# Kernel formula:
# w_i = exp(-d_i^2 / sigma^2)
#
# Breakdown:
# - w_i    -> weight of the i-th neighbour (perturbation) — this is the output,
#             the thing we're computing
# - d_i    -> distance of the i-th neighbour from the original instance x
#             (Sampler already gives us d_i^2 directly in dist_sq_perturbations[i],
#             so no need to square it again here)
# - sigma  -> kernel width hyperparameter, passed in from outside
# !- exp()  -> the exponential function (e^x); squashes the result into (0, 1]
# - the "-" sign -> makes weight decrease as distance increases
#
# NumPy equivalent (vectorized, all neighbours at once):
# weights = np.exp(-np.array(dist_sq_perturbations) / sigma**2)
#
# Note: dist_sq_perturbations is already "distance squared" (d_i^2), computed
# in Sampler as distance_sq = np.sum((x - neighbour)**2). So there's no extra
# **2 needed here — it's already squared.