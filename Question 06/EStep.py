import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####

     N, D = X.shape
    # K = weights.shape[1]
    K = np.array(weights).reshape(1,-1).shape[1]
    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    gamma =np.zeros((N,K))

    for n in range(N):
        gamma_nominator = np.zeros((1,K))
        
        gamma_denom =0
        for k in range(K):
            delta = X[n,:]-means[k]
            cov_mat = covariances[:,:,k]
            cov_mat_inv = np.linalg.pinv(cov_mat)
            denominator = (((2*np.pi)**(D/2))* np.sqrt(np.linalg.det(cov_mat)))
            gamma_nominator[0,k] = np.array(weights).reshape(1,-1)[0,k]*np.exp(-0.5*(delta.T @ (cov_mat_inv @ delta)))/denominator
            gamma_denom += gamma_nominator[0,k]
        for k in range(K):
            gamma[n, k] = gamma_nominator[0,k]/gamma_denom
            
            
        # print("the value of gamma is: ",gamma.shape)
    return [logLikelihood, gamma]
