import pandas as pd
import numpy as np
from surprise import SVD
from surprise import GridSearch
from surprise import KNNBaseline
from surprise import BaselineOnly
from surprise import evaluate
import xlsxwriter

# Here we use the grid search algo from surprise and return the best elems
def grid_search_als_surprise(data_train, n_epochs, reg_u, reg_i):
    print('ALS Surprise grid search')

    #Construct ALS algo with params
    param_grid = {'bsl_options': {'method': ['als'],
                                  'n_epochs' : n_epochs,
                                  'reg_u': reg_u,
                                  'reg_i': reg_i},
                  'k': [3],
                  'sim_options': {'name': ['msd'],
                                  'min_support': [5],
                                  'user_based': [False]}
                  }

    #Create the Grid search algo with the params
    grid_search = GridSearch(KNNBaseline, param_grid, measures=['RMSE'])
    #Evaluate the Grid search and print the best params
    grid_search.evaluate(data_train)
    print(grid_search.best_score['RMSE'])
    print(grid_search.best_params['RMSE'])
    # Return the best params for the ALS algo
    return grid_search.best_params['RMSE']["bsl_options"]

# Here we use the ALS algo manually. We compute manually the the algo and store the values
def grid_search_als(data_train, data_test, n_epochs, reg_us, reg_is, file_name):

    print('ALS Surprise manual grid search')

    result_train = pd.DataFrame()
    result_test = pd.DataFrame()

    # loops on the parameters
    for n_epoch in n_epochs:
        for reg_u in reg_us:
            for reg_i in reg_is:


                bsl_options = {'method': 'als',
                                'n_epochs': n_epoch,
                                'reg_u': reg_u,
                                'reg_i': reg_i
                            }

                algo = BaselineOnly(bsl_options=bsl_options)
                # Retrieve the trainset.
                trainset = data_train.build_full_trainset()

                # Build an algorithm, and train it.
                algo.train(trainset)
                #Evaluate the performance
                perf_train = evaluate(algo, data_train, measures=['RMSE'])
                perf_test = evaluate(algo, data_test, measures=['RMSE'])

                perf_train["n_epoch"] = n_epoch
                perf_train["reg_u"] = reg_u
                perf_train["reg_i"] = reg_i
                #Store the mean performance RMSE on train
                perf_train["rmse"] = np.mean(perf_train['rmse'])

                perf_test["n_epoch"] = n_epoch
                perf_test["reg_u"] = reg_u
                perf_test["reg_i"] = reg_i
                #Store the mean performance RMSE on test
                perf_test["rmse"] = np.mean(perf_test['rmse'])

                #Store on a dataframe
                result_train = result_train.append(perf_train, ignore_index=True)
                result_test = result_test.append(perf_test, ignore_index=True)

    # Save the dataframe so we will see or plot the differencies if it's interesting
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    result_train.to_excel(writer, 'Sheet1')
    result_test.to_excel(writer, 'Sheet2')
    writer.save()

# Here we use the grid search algo from surprise and return the best elems
def grid_search_svd_surprise(data_train, n_epochs, lr_all, reg_all, init_mean, n_factors):

    print('SVD Surprise grid search')

    # Construct SVD algo with params
    param_grid = {'n_epochs': n_epochs, 'lr_all': lr_all, 'reg_all': reg_all, 'init_mean': init_mean, 'n_factors':n_factors}
    # Create the Grid search algo with the params
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'],verbose=False)
    # Evaluate the Grid search and print the best params
    grid_search.evaluate(data_train)
    print(grid_search.best_score['RMSE'])
    print(grid_search.best_params['RMSE'])
    # Return the best params for the SVD algo
    return grid_search.best_params['RMSE']

# Here we use the SVD algo manually. We compute manually the the algo and store the values
def grid_search_svd(data_train,data_test, n_epochs, lr_alls, reg_alls, init_mean, n_factors, file_name):

    print('SVD Surprise manual grid search')

    result_train = pd.DataFrame()
    result_test = pd.DataFrame()
    # loops on the parameters
    for n_epoch in n_epochs:
        for lr_all in lr_alls:
            for reg_all in reg_alls:
                for n_factor in n_factors:

                    algo = SVD(reg_all=reg_all, init_mean=init_mean, n_epochs=n_epoch, lr_all=lr_all, n_factors=n_factor)

                    # Retrieve the trainset.
                    trainset = data_train.build_full_trainset()

                    # Build an algorithm, and train it.
                    algo.train(trainset)
                    # Evaluate the performance
                    perf_train = evaluate(algo, data_train,measures=['RMSE'])
                    perf_test = evaluate(algo, data_test,measures=['RMSE'])

                    perf_train["n_epoch"] = n_epoch
                    perf_train["lr_all"] = lr_all
                    perf_train["reg_all"] = reg_all
                    perf_train["init_mean"] = init_mean
                    perf_train["n_factor"] = n_factor
                    # Store the mean performance RMSE on train
                    perf_train["rmse"] = np.mean(perf_train['rmse'])

                    perf_test["n_epoch"] = n_epoch
                    perf_test["lr_all"] = lr_all
                    perf_test["reg_all"] = reg_all
                    perf_test["init_mean"] = init_mean
                    perf_test["n_factor"] = n_factor
                    # Store the mean performance RMSE on test
                    perf_test["rmse"] = np.mean(perf_test['rmse'])

                    # Store on a dataframe
                    result_train = result_train.append(perf_train, ignore_index=True)
                    result_test = result_test.append(perf_test, ignore_index=True)

    # Save the dataframe so we will see or plot the differencies if it's interesting
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    result_train.to_excel(writer, 'Sheet1')
    result_test.to_excel(writer, 'Sheet2')
    writer.save()
