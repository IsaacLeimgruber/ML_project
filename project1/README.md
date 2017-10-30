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
 
### Project file
    Our Project test file is divided in 7 parts

    1) First we loaded the data and add a function compare_pred to compare the training data and see if our ratio was good.

    2) Least squares (Just a basic least square to see the prediction that we have)

    3) Stochastic Gradient Descent (to see if it works and if it can be used after)

    4) Ridge Regression: 
            In this part we tried several things, first of all we tried our Ridge 
            algo and find the best parameters with a grid search algorithm (func-  
            tion "ridge_regression_demo"). When we found it, the result was better  
            than the least square so we submitted the result on kaggle. However we  
            thought that with a cross validation, our result could be improved so   
            with the function  
            "cross_val_ridge_demo"which find the best parameters and  
            "cross_val_ridge" for splitting the data and compute our  
            weights and loss everytime. We added a   "cross_valid_vizu"   
            function to plot the result and understand them better. We found  
            out that the best degree gives better result and  
            we decided to take this degree (despite the fact that on the plot we can  
            see that it could explode the final result)  
            However it didn't improve our result so we tought   
            that the grid search worked better with the Ridge regression'  
            At the end we found out that if we deleted some columns we could   
            improve our result but it was too late. We added this part  
            to see our (little) improvement at the end of this section  
    5) Logistic regression:
            First of all we added a function "shuffle_index_resize". This function, as the name say shuffle the index and resize our x_train and y_train. Why? this is for doing a better Stochastic gradient descent. We saw that the values at the end where -1 and 1. So st the end we choose a number of elems (same in both list) and take them. With this we will have the same amount of 1 and -1 and with this we thought that the result would be better.
            We added a function "compare_prediction2" because we saw that the sigmoid gave values between 0 and 1 and not between -1 and 1. However we decided to change the sigmoid so the values returned will be between -1 and 1 and not 0 and 1.

            After that we have the cross validation for the Logistic regression. We decided to implement the Logistic penalized regression because otherwise we will have error due to overfitting.
            In the end of this section there are the plots added on the reports to visualise our results

We saw that the logistic part wasn't the best part so we decided to improve the ridge regression by loking on the data and manipulate them to have better result. That's why the next section is called New Ridge


    6) New Ridge:
            In this Section we tried so split the data because we found out that one col (PRI_jet_num) indicate some groupe of data and we thought that if we group the data and find the best w for each group, our result would be improved. For this, we first took the indexes of the data where the col PRI-jet_num was different to partitionate the data. We didn't needed this column anymore so we deleted it. Secondly we add function "bad_values_in_zero" and "bad_values_in_mean" because some values where at -999 and this could penalise to much our result. So with these functions we changed the values by 0 or the mean to see the differencies (not very good but discussed later). 
            After that we decided to implement a "ridge_Grid_search". Why? Because the best result found previously where with the grid search the result was better. At this point we iterated over the 4 groups to find the best weight.
            Now that we have the weights, we need to implement a new function "build_prediction". It will take data x and applied the right weight depending on which group they were. Unfortunately at the end the result was very poor so we think that we missed something or our parameters weren't the best

    7) Cross Validation new Ridge:
            We found out previously that the final result weren't good. So to be sure of our parameters, we decided to do a cross validation and plot the result to understand what we did wrong. What seems strange, is that our graphs are pretty good but give a result very poor. You can see the plots on the last part of the file. We didn't added to the report because our assumptions have unfortunately proved wrong. We were sure that with this technic, our result would be improved but it wasn't the case.'
