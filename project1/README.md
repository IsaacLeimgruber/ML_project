## A machine learning implementation to predict higgs boson appearance from a collection of events

### How to run
  You don't need any specific package except the numpy library and standard python library.
  
  You can reproduce our result by running the run.py script
  
### Helper files
  implementations.py: 5 functions we needed to implement: least squares, (stochastic) gradient descent, ridge regression, 
      logistic (penalized) regression
     
  batch_iter.py: contains the batch_iter function for the stochastic gradient descent
  
  build_k_indices.py: partitions the data indices in k partitions for the cross-validation
  
  build_polynomial.py: build a polynomial model from the data, ie the phi function phi(x) = 1 + x + x^2 +...+x^n
  
  gradient.py: computes gradient for different regressions
  
  learning_by_gradient_descent.py: implements the gradient descent regression
  
  learning_by_penalized_gradient.py: '--------------------------------------' penalized version 
  
  loss.py: compute_loss for ridge and least squares, calculate_loss for logistic regression
  
  penalized_logistic_regression.py: computes loss and gradient for penalized logistic regression
  
  sigmoid.py: implementation of the sigmoid function [0,1]
  
  split_data.py: shuffles the data and split it into training data and testing data
  
  project1.ipynb: main file containing tests and attempts to obtain the best predictions
 
###
  
  
