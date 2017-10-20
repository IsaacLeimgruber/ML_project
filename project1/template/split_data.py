# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)

    # split the data based on the given ratio

    idx = [i for i in range(len(x))]
    np.random.shuffle(idx)
    split = int(len(x) * ratio)
    
    x_shuffle = x[idx]
    y_shuffle = y[idx]
    
    x_train = x_shuffle[:split]
    x_test = x_shuffle[split:]
    y_train = y_shuffle[:split]
    y_test = y_shuffle[split:]
    
    return x_train, y_train, x_test, y_test 
