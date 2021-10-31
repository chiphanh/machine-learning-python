import numpy as np
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    N, D = shape(X)
    K = shape(weights)[1]
    logLikelihood =0
    for n in range(N):
        prob = 0
        for k in range(K):
            delta = X[n,:]-means[k]
            cov_mat = covariances[:,:,k]
            cov_mat_inv = np.linalg.inv(cov_mat)
            denominator = (((2*np.pi)**(D/2))* np.sqrt(np.linalg.det(cov_mat)))
            prob += weights[1,k]*np.exp(-0.5*(delta.T @ (cov_mat_inv @ delta)))/denominator
        logLikelihood += np.log(prob)
        
    return logLikelihood

