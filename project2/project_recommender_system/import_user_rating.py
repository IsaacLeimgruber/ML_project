import numpy as np
import pandas as pd
from my_helpers import *
from ALS_surprise import als_surprise
from SVD_surprise import svd_surprise
from surprise import Reader
from surprise import Dataset
import xlrd

DATA_PATH_10 = "keep_users_bigger_10_rating.xlsx"
DATA_PATH_20 = "keep_users_bigger_20_rating.xlsx"
DATA_PATH_50 = "keep_users_bigger_50_rating.xlsx"


def main():

    for i in range(1):

        if(i==0):
            df_data = pd.read_excel(DATA_PATH_10, index_col=0, header=0)
        elif(i == 1):
            df_data = pd.read_excel(DATA_PATH_20, index_col=0, header=0)
        else:
            df_data = pd.read_excel(DATA_PATH_50, index_col=0, header=0)


        userId = df_data["userId"]

        movieId = df_data["movieId"]
        rating = df_data["rating"]

        ratings_dict = {'itemID': movieId,
                        'userID': userId,
                        'rating': rating}

        df = pd.DataFrame(ratings_dict)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        data.split(3)

        perf_als_grid = grid_search_als(data)

        perf_als, predictions_als = als_surprise(df, perf_als_grid["n_epochs"], perf_als_grid["reg_u"],
                                                 perf_als_grid["reg_i"], 0)


        perf_als_grid = grid_search_svd(data)

        # svd_surprise(data, reg_all, init_mean, n_epochs, lr_all)
        perf, predictions_svd = svd_surprise(df, perf_als_grid["reg_all"], perf_als_grid["init_mean"],
                                             perf_als_grid["n_epochs"], perf_als_grid["lr_all"],
                                             perf_als_grid["n_factors"])




        if(i == 0):
            make_prediction(predictions_als, "ALS_bigger_10_user.csv")
            make_prediction(predictions_svd, "SVD_bigger_10_user.csv")
        elif(i == 1):
            make_prediction(predictions_als, "ALS_bigger_20_user.csv")
            make_prediction(predictions_svd, "SVD_bigger_20_user.csv")
        else:
            make_prediction(predictions_als, "ALS_bigger_50_user.csv")
            make_prediction(predictions_svd, "SVD_bigger_50_user.csv")



if __name__ == '__main__':
    main()
