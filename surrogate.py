# β = (XᵀWX + λI)⁻¹ XᵀWy
import numpy as np
class LocalSurrogate:
    def __init__(self, masks, y, weights, lam):
        """
    Args:
    masks (list of np.ndarray): Sampler's self.masks list — becomes the 
        basis for the X matrix. Currently a Python list; convert with 
        np.array(masks) into an (n, d) shaped matrix (n = number of 
        samples, d = number of features).
    y (np.ndarray): the real model's prediction for each neighbour — 
        results of calling predict_fn(perturbations[i]). Not available 
        yet at this stage (comes in with LimeExplainer later), so for 
        testing you can generate a fake y array by hand (e.g. 
        np.random.rand(n)).
    weights (np.ndarray): Kernel's self.weights — shape (n,).
    lam (float): ridge regularization coefficient, a small constant 
        (e.g. 0.01).
    """
    
        self.masks = masks
        self.y = y
        self.weights = weights
        self.lam = lam

        X = np.array(self.masks) # convert 2 np & STACK
        X = X.reshape(X.shape[0], -1) # UNSTACK 
        bias_col = np.ones((X.shape[0], 1))
        X = np.hstack([bias_col, X]) # merge with bias coulmn
        
        # X = sample_size x (feature_num + 1(bias coulmn) )
        
        W = np.diag(self.weights)
        
        # A = XᵀWX + λI
        # B = XᵀWy
        # beta = A⁻¹ @ B   
        
        A = (X.T @ W @ X) + self.lam * np.eye(X.shape[1]) # ridge regulazation on lam term as a basic complexity penalty function
        B = (X.T @ W @ self.y)
        # look parantehesis similarity
        self.beta = np.linalg.solve(A, B) # A @ beta = B
        #self.beta = [beta_0, beta_1, beta_2, beta_3, ..., beta_150]
            #            ^bias    ^feature 0  ^feature 1   ...   ^feature 149
        
        # y ≈ beta[0] + beta[1]*feature_1 + beta[2]*feature_2 + ... + beta[d]*feature_d
        #! beta[0] is the BİAS that prevents if all features 0 = "0" output case