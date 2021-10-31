import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####

    N,D = X.shape
    K= gamma.shape[1]
    # soft number of samples labeled k:
    N_k = gamma.sum(axis = 0).reshape(1,-1) #shape (1,K)
    #weights:
    weights = N_k/N #shape (1,K)
    #new means:
    means = (gamma.T @ X)/(N_k.T )  #shape (KxD)
    # new covariances
    covariances = np.zeros((D,D,K))
    for k in range(K):
        sigma = np.zeros((D,D))
        for n in range(N):
            new_delta = X[n,:]-means[k,:]
            sigma += gamma(n,k)*(new_delta @ new_delta.T)
        covariances[:,:,k] = sigma/N_k[0,k]
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return weights, means, covariances, logLikelihood
