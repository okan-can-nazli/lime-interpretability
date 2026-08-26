import numpy as np

class SAMPLER:
    
    def __init__(self, x, r, n, noise_frac):
        """_

        Args:
            x (input data matrix)
            r (range)
            n (sample number)
        """
        self.x = x
        self.r = r
        self.n = n
        
        self.noise_frac = noise_frac # distribution rate of the instances
        self.perturbations = []
        self.dist_sq_perturbations = []
        
        max_tries = n * 1000 # prevent infinite loop bug
        tries = 0
        while(len(self.perturbations) < n):
            
            tries += 1
            
            if tries > max_tries:
                raise ValueError(f"Only found {len(self.perturbations)}/{n} valid perturbations after {tries} tries. r={r} too small or noise_frac={noise_frac} too large.")


            neighbour = x + np.random.randn(*x.shape) * min(noise_frac, r / np.sqrt(x.size)) # distribute with given noise_frac OR reduce it to handle
            distance_sq = np.sum((x - neighbour)**2)

            # what if noise is too much? İNFİNİTE LOOP ,no found perturbation
            if (distance_sq < r**2):
                self.perturbations.append(neighbour)
                self.dist_sq_perturbations.append(distance_sq) # save perturbation dist
                
            
        

        
        
