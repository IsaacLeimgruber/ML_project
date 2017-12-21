import numpy as np
import pandas as pd
from my_helpers import *
from surprise_cross_validation import *
from surprise_knn import knn_surprise
from surprise_svd import svd_surprise
from surprise import Reader
from surprise import Dataset
import xlrd

DATA_PATH = "data_train.csv"
DATA_PATH_10 = "users_movies_bigger_10_rating.xlsx"
DATA_PATH_20 = "users_movies_bigger_20_rating.xlsx"
DATA_PATH_50 = "users_movies_bigger_50_rating.xlsx"
DATA_PATH_SUB = "sample_submission.csv"


def main():
    #Here we have 4 data that can be taken for the test. The basic one, the others with users and movies that have more than 10, 20 and 50 ratings
    for i in range(4):
        if(i == 0):
            #Import the basic data
            data_import = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1, dtype=str)
            #Construct User, Movie and rating
            userId, movieId, rating = construct_data(data_import);
        elif(i == 1):
            # Import the data with users and movies that have more than 10 ratings
            data_import = pd.read_excel(DATA_PATH_10, index_col=0, header=0)
            userId = data_import["userId"]
            movieId = data_import["movieId"]
            rating = data_import["rating"]
        elif(i == 2):
            # Import the data with users and movies that have more than 20 ratings
            data_import = pd.read_excel(DATA_PATH_20, index_col=0, header=0)
            userId = data_import["userId"]
            movieId = data_import["movieId"]
            rating = data_import["rating"]
        else:
            # Import the data with users and movies that have more than 30 ratings
            data_import = pd.read_excel(DATA_PATH_50, index_col=0, header=0)
            userId = data_import["userId"]
            movieId = data_import["movieId"]
            rating = data_import["rating"]

        #We take the indicies that we will shuffle
        indices_to_shuffle = np.array(range(len(userId)))

        test_ratio = 70
        #We create the train and test data (70% ratio of the data does on train) Indicies are shuffeled in split_data
        X_train_userId, X_train_movieId, X_train_rating, X_test_userId, X_test_movieId, X_test_rating = split_data(
            indices_to_shuffle, userId, movieId, rating, test_ratio)

        ratings_dict_train = {'itemID': X_train_movieId,
                        'userID': X_train_userId,
                        'rating': X_train_rating}

        #Create the dataframe for the surprise train
        df_train = pd.DataFrame(ratings_dict_train)
        reader_train = Reader(rating_scale=(1, 5))
        data_train = Dataset.load_from_df(df_train[['userID', 'itemID', 'rating']], reader_train)

        #We have to split the data because the algo test with the splited elements
        split = 3
        data_train.split(split)

        ratings_dict_train = {'itemID': X_test_movieId,
                            'userID': X_test_userId,
                            'rating': X_test_rating}

        # Create the dataframe for the surprise test
        df_test = pd.DataFrame(ratings_dict_train)
        reader_test = Reader(rating_scale=(1, 5))
        data_test = Dataset.load_from_df(df_test[['userID', 'itemID', 'rating']], reader_test)
        data_test.split(split)

        #KNN (takes to long to test all of them but you can check)
        #n_epochs = [5,10,15]
        #reg_us = [5,10,15,20]
        #reg_is = [5,10,20]

        # KNN best param
        n_epochs = [5]
        reg_us = [5]
        reg_is = [5]

        # Apply the grid search for KNN
        perf_als_grid = grid_search_als_surprise(data_train, n_epochs, reg_us, reg_is)

        if (i == 0):
            #Manual grid search so we can see the values (only the result is given with the GridSearch of surprise)
            grid_search_als(data_train, data_test, n_epochs, reg_us, reg_is, 'surpise_manualGS_KNN.xlsx')
            # KNN with the best params from GridSearch surprise
            knn_surprise(data_train, perf_als_grid["n_epochs"], perf_als_grid["reg_u"], perf_als_grid["reg_i"], "surprise_bestKNN.csv")

        elif(i == 1):
            # Manual grid search so we can see the values (only the result is given with the GridSearch of surprise)
            grid_search_als(data_train, data_test, n_epochs, reg_us, reg_is, 'surpise_manualGS_KNN10.xlsx')
            # KNN with the best params from GridSearch surprise
            knn_surprise(data_train, perf_als_grid["n_epochs"], perf_als_grid["reg_u"], perf_als_grid["reg_i"],"surprise_bestKNN10.csv")
        elif (i == 2):
            # Manual grid search so we can see the values (only the result is given with the GridSearch of surprise)
            grid_search_als(data_train, data_test, n_epochs, reg_us, reg_is, 'surpise_manualGS_KNN20.xlsx')
            # KNN with the best params from GridSearch surprise
            knn_surprise(data_train, perf_als_grid["n_epochs"], perf_als_grid["reg_u"], perf_als_grid["reg_i"],"surprise_bestKNN20.csv")
        else:
            # Manual grid search so we can see the values (only the result is given with the GridSearch of surprise)
            grid_search_als(data_train, data_test, n_epochs, reg_us, reg_is, 'surpise_manualGS_KNN50.xlsx')
            # KNN with the best params from GridSearch surprise
            knn_surprise(data_train, perf_als_grid["n_epochs"], perf_als_grid["reg_u"], perf_als_grid["reg_i"],"surprise_bestKNN50.csv")

        #SVD (takes to long to test all of them but you can check)
        #n_epochs = [5,10]
        #lr_alls = [0.00145, 0.00146, 0.00147]
        #reg_alls = [0.2,0.3]
        #init_mean = [0, 0.2]
        #n_factors = [80,100,120]

        # SVD best param
        n_epochs = [10]
        lr_alls = [0.00147]
        reg_alls = [0.2]
        init_mean = [0.2]
        n_factors = [80]

        # Apply the grid search for SVD
        perf_svd_grid = grid_search_svd_surprise(data_train, n_epochs, lr_alls, reg_alls, init_mean, n_factors)

        if (i == 0):
            # Manual grid search so we can see the values (only the result is given with the GridSearch of surprise)
            grid_search_svd(data_train, data_test, n_epochs, lr_alls, reg_alls, init_mean, n_factors,
                            'surpise_manualGS_SVD.xlsx')
            # SVD with the best params from GridSearch surprise
            svd_surprise(data_train, perf_svd_grid["reg_all"], perf_svd_grid["init_mean"], perf_svd_grid["n_epochs"],
                         perf_svd_grid["lr_all"], perf_svd_grid["n_factors"], "surprise_bestSVD.csv")

        elif(i == 1):
            # Manual grid search so we can see the values (only the result is given with the GridSearch of surprise)
            grid_search_svd(data_train, data_test, n_epochs, lr_alls, reg_alls, init_mean, n_factors,
                            'surpise_manualGS_SVD10.xlsx')
            # SVD with the best params from GridSearch surprise
            svd_surprise(data_train, perf_svd_grid["reg_all"], perf_svd_grid["init_mean"], perf_svd_grid["n_epochs"],
                         perf_svd_grid["lr_all"], perf_svd_grid["n_factors"], "surprise_bestSVD10.csv")

        elif (i == 2):
            # Manual grid search so we can see the values (only the result is given with the GridSearch of surprise)
            grid_search_svd(data_train, data_test, n_epochs, lr_alls, reg_alls, init_mean, n_factors,
                            'surpise_manualGS_SVD20.xlsx')
            # SVD with the best params from GridSearch surprise
            svd_surprise(data_train, perf_svd_grid["reg_all"], perf_svd_grid["init_mean"], perf_svd_grid["n_epochs"],
                         perf_svd_grid["lr_all"], perf_svd_grid["n_factors"], "surprise_bestSVD20.csv")

        else:
            # Manual grid search so we can see the values (only the result is given with the GridSearch of surprise)
            grid_search_svd(data_train, data_test, n_epochs, lr_alls, reg_alls, init_mean, n_factors,
                            'surpise_manualGS_SVD50.xlsx')
            # SVD with the best params from GridSearch surprise
            svd_surprise(data_train, perf_svd_grid["reg_all"], perf_svd_grid["init_mean"], perf_svd_grid["n_epochs"],
                         perf_svd_grid["lr_all"], perf_svd_grid["n_factors"], "surprise_bestSVD50.csv")


if __name__ == '__main__':
    main()
