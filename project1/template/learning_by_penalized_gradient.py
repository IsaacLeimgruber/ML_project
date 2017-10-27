import numpy as np
from penalized_logistic_regression import *

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):

    # return loss, gradient
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    # update w
    
    w = w - gamma * grad
    
    return loss, w
