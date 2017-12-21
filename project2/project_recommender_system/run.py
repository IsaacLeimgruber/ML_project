import numpy as np
import scipy.sparse as csr
from random import shuffle
from ALS import run_ALS
from plots import plot_raw_data
from helpers import *
from my_helpers import *
import scipy.sparse as sp
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# https://www.overleaf.com/12522751fsgrdpydrwxf READ and EDIT link for report
DATA_PATH = "data_train.csv"
DATA_PATH_SUB = "sample_submission.csv"


def main():
    data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1, dtype=str)
    sub_data = np.genfromtxt(DATA_PATH_SUB, delimiter=",", skip_header=1, dtype=str)

    userId, movieId, rating = construct_data(data);

    indices_to_shuffle = np.array(range(len(userId)))
    print("test")

    X_train_userId, X_train_movieId, X_train_rating, X_test_userId, X_test_movieId, X_test_rating = split_data(indices_to_shuffle, userId, movieId, rating, 70)

    data_XTrain = csr.csr_matrix((X_train_rating, (X_train_userId, X_train_movieId)), shape=(10000, 1000))
    data_XTest = csr.csr_matrix((X_test_rating, (X_test_userId, X_test_movieId)), shape=(10000, 1000))
    #print((data_XTrain.T).index(928))
    # attention no movie 928?

    print("TEST ALS")

    #rmse, users, items = run_ALS(data_XTrain, data_XTest, 20, 0.1, 0.7)
    rmse, users, items = run_ALS(data_XTrain, data_XTest, 20, 31.8553, 20.05672522)
    print("rmse als", rmse)
    print("rmse users", np.shape(users))
    print("rmse items", np.shape(items))


    #prediction = []
    #for iRow in sub_data:
    #    user, movie, rating = parse_row(iRow)
    #    rate = int(np.round(avgMovie[movie]))
    #    #rate = avgGlobal
    #    prediction.append([user+1, movie+1, rate])

    #create_submission(prediction, "GLOBAL_MOVIE_AVG.csv")
    #create_submission(prediction, "GLOBAL_AVG.csv")


if __name__ == '__main__':
    main()