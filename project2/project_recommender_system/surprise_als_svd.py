import numpy as np
import pandas as pd
from my_helpers import *
from cross_validation_surprise import *
from ALS_surprise import als_surprise
from SVD_surprise import svd_surprise
from surprise import BaselineOnly
from surprise import Reader
from surprise import Dataset
import xlrd

DATA_PATH = "data_train.csv"
DATA_PATH_SUB = "sample_submission.csv"
DATA_PATH_10 = "keep_users_bigger_10_rating.xlsx"
DATA_PATH_20 = "keep_users_bigger_20_rating.xlsx"
DATA_PATH_50 = "keep_users_bigger_50_rating.xlsx"


def main():
    for i in range(4):
        if(i == 0):
            data_import = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1, dtype=str)
            userId, movieId, rating = construct_data(data_import);
        elif(i == 1):
            data_import = pd.read_excel(DATA_PATH_10, index_col=0, header=0)
            userId = data_import["userId"]
            movieId = data_import["movieId"]
            rating = data_import["rating"]
        elif(i == 2):
            data_import = pd.read_excel(DATA_PATH_20, index_col=0, header=0)
            userId = data_import["userId"]
            movieId = data_import["movieId"]
            rating = data_import["rating"]
        else:
            data_import = pd.read_excel(DATA_PATH_50, index_col=0, header=0)
            userId = data_import["userId"]
            movieId = data_import["movieId"]
            rating = data_import["rating"]


        indices_to_shuffle = np.array(range(len(userId)))

        test_ratio = 70
        X_train_userId, X_train_movieId, X_train_rating, X_test_userId, X_test_movieId, X_test_rating = split_data(
            indices_to_shuffle, userId, movieId, rating, test_ratio)

        ratings_dict_train = {'itemID': X_train_movieId,
                        'userID': X_train_userId,
                        'rating': X_train_rating}

        df_train = pd.DataFrame(ratings_dict_train)
        reader_train = Reader(rating_scale=(1, 5))
        data_train = Dataset.load_from_df(df_train[['userID', 'itemID', 'rating']], reader_train)

        split = 3
        data_train.split(split)

        ratings_dict_train = {'itemID': X_test_movieId,
                            'userID': X_test_userId,
                            'rating': X_test_rating}

        df_test = pd.DataFrame(ratings_dict_train)
        reader_test = Reader(rating_scale=(1, 5))
        data_test = Dataset.load_from_df(df_test[['userID', 'itemID', 'rating']], reader_test)
        data_test.split(split)

        #ALS
        n_epochs = [5,10,20]
        reg_us = [10,15,20]
        reg_is = [5,10,20]

        #n_epochs = [10]
        #reg_us = [15]
        #reg_is = [10]
    
        perf_als_grid = grid_search_als_surprise(data_train, n_epochs, reg_us, reg_is)
        perf_als, predictions_als = als_surprise(data_train, perf_als_grid["n_epochs"], perf_als_grid["reg_u"],perf_als_grid["reg_i"])

        if (i == 0):
            grid_search_als(data_train, data_test, n_epochs, reg_us, reg_is, 'value_manual_ALS_grid_search.xlsx')
            make_prediction(predictions_als, "ALS_best.csv")
        elif(i == 1):
            grid_search_als(data_train, data_test, n_epochs, reg_us, reg_is, 'value_manual_ALS10_grid_search.xlsx')
            make_prediction(predictions_als, "ALS_10best.csv")
        elif (i == 2):
            grid_search_als(data_train, data_test, n_epochs, reg_us, reg_is, 'value_manual_ALS20_grid_search.xlsx')
            make_prediction(predictions_als, "ALS_20best.csv")
        else:
            grid_search_als(data_train, data_test, n_epochs, reg_us, reg_is, 'value_manual_ALS50_grid_search.xlsx')
            make_prediction(predictions_als, "ALS_50best.csv")

        #SVD
        n_epochs = [5,10]
        lr_alls = [0.00145, 0.00146, 0.00147]
        #lr_alls = [0.00147]
        reg_alls = [0.2,0.3]
        init_mean = [0.2]
        n_factors = [80,100,120]

        perf_svd_grid = grid_search_svd_surprise(data_train, n_epochs, lr_alls, reg_alls, init_mean, n_factors)
        perf_svd, predictions_svd = svd_surprise(data_train, perf_svd_grid["reg_all"], perf_svd_grid["init_mean"],
                                         perf_svd_grid["n_epochs"], perf_svd_grid["lr_all"], perf_svd_grid["n_factors"])


        if (i == 0):
            grid_search_svd(data_train, data_test, n_epochs, lr_alls, reg_alls, init_mean, n_factors,
                            'value_manual_SVD_grid_search.xlsx')
            make_prediction(predictions_svd, "SVD_best.csv")
        elif(i == 1):
            grid_search_svd(data_train, data_test, n_epochs, lr_alls, reg_alls, init_mean, n_factors,
                            'value_manual_SVD10_grid_search.xlsx')
            make_prediction(predictions_svd, "SVD10_best.csv")
        elif (i == 2):
            grid_search_svd(data_train, data_test, n_epochs, lr_alls, reg_alls, init_mean, n_factors,
                            'value_manual_SVD20_grid_search.xlsx')
            make_prediction(predictions_svd, "SVD20_best.csv")
        else:
            grid_search_svd(data_train, data_test, n_epochs, lr_alls, reg_alls, init_mean, n_factors,
                            'value_manual_SVD50_grid_search.xlsx')
            make_prediction(predictions_svd, "SVD50_best.csv")


if __name__ == '__main__':
    main()
