# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

stochastic_batch_size = 10;

def compute_loss(y, tx, w):
    e = y - tx @ w
    return np.mean(e**2)/2.
   

def compute_gradient(y, tx, w):
    e = y - tx @ w
    grad = -np.transpose(tx)@e/len(e)
    return grad;

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    # regarder que le loss min
    loss = 1000;
    w = initial_w
    for n_iter in range(max_iters):
        
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
    
    return w, loss;


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)
    
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices] 
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e)
    return grad/len(e);

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    loss = 0;
    for n_iter in range(max_iters):

        for minibatch_y, minibatch_tx in batch_iter(y, tx, stochastic_batch_size):
            grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)
            
    return w, loss;

def least_squares(y, tx):
    """calculate the least squares solution."""
    
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss(y, tx, w)
    return w, loss;

def ridge_regression(y, tx, lambda_):

    first_term = tx.T@tx
    #sum ici les x*w
    left = first_term + lambda_ *np.identity(tx.shape[1])
    right = tx.T @ y
    w = np.linalg.solve(left, right)
    loss = compute_loss(y, tx, w)
    return w, loss;

