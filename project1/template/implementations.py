# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

from loss import *
from gradient import *
from batch_iter import *
from learning_by_gradient_descent import *
from learning_by_penalized_gradient import *
from split_data import *


stochastic_batch_size = 10;


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


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print("shit")
            break
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss
