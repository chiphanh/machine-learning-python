import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)

    m = data.shape[0]
    n = data.shape[1]
    x0 = np.ones((m,1))
    X = np.hstack((x0, data))
    weight_vec = np.zeros((n+1,1))
    weight_vec = np.dot(np.linalg.pinv(X), label)
    weight = weight_vec[1:].reshape(-1,1)
    bias = weight_vec[0]


    return weight, bias
