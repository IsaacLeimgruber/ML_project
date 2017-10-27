import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    #sig = (np.exp(t) / (1 + np.exp(t))) # this is for values between 0 and 1
    sig = 2*(np.exp(t) / (1 + np.exp(t))) - 1 # this is for values between -1 and 1 better for us
    return sig
