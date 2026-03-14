import numpy as np


def gaussian_pdf(x, mean, cov):
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    norm_const = 1.0 / (2 * np.pi * np.sqrt(det_cov))
    diff = x - mean
    exponent = -0.5 * diff @ inv_cov @ diff
    return norm_const * np.exp(exponent)

def gmm_pdf(x, means, covs, weights):
    """Evaluate Gaussian mixture density at points x."""
    probs = np.zeros(x.shape[0])
    for k in range(len(means)):
        diff = x - means[k]
        inv_cov = np.linalg.inv(covs[k])
        det_cov = np.linalg.det(covs[k])
        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** 2 * det_cov))
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        probs += weights[k] * norm_const * np.exp(exponent)
    return probs

def score_function(points, means, covs, weights):
    """
    Compute the score (∇ log p(x)) for a batch of points in R^2.
    points: array of shape (N, 2)
    Returns: array of shape (N, 2)
    """
    N = points.shape[0]
    scores = np.zeros((N, 2))
    inv_covs = np.linalg.inv(covs)
    
    for i, x in enumerate(points):
        # Compute component densities
        pdfs = np.array([
            weights[k] * gaussian_pdf(x, means[k], covs[k])
            for k in range(2)
        ])
        denom = np.sum(pdfs)
        
        # Responsibilities
        r = pdfs / denom
        
        # Weighted sum of component scores
        score = np.zeros(2)
        for k in range(2):
            score += r[k] * (inv_covs[k] @ (means[k] - x))
        
        scores[i] = score
    
    return scores