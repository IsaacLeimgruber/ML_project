import numpy as np
import pandas as pd
from my_helpers import *
from cross_validation import *
from ALS_surprise import als_surprise
from SVD_surprise import svd_surprise
from surprise import Reader
from surprise import Dataset


# https://www.overleaf.com/12522751fsgrdpydrwxf READ and EDIT link for report

DATA_PATH = "data_train.csv"
DATA_PATH_SUB = "sample_submission.csv"


def main():
    data_import = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1, dtype=str)
    userId, movieId, rating = construct_data(data_import);

    ratings_dict = {'itemID': movieId,
                    'userID': userId,
                    'rating': rating}

    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    data.split(3)

    perf_als_grid = grid_search_als(data)

    perf_als, predictions_als = als_surprise(df, perf_als_grid["n_epochs"], perf_als_grid["reg_u"], perf_als_grid["reg_i"], 0)

    make_prediction(predictions_als, "ALS_best.csv")



    perf_als_grid = grid_search_svd(data)

    #svd_surprise(data, reg_all, init_mean, n_epochs, lr_all)
    perf, predictions_svd = svd_surprise(df, perf_als_grid["reg_all"], perf_als_grid["init_mean"], perf_als_grid["n_epochs"], perf_als_grid["lr_all"], perf_als_grid["n_factors"])

    make_prediction(predictions_svd, "SVD_best.csv")



if __name__ == '__main__':
    main()
