import numpy as np
import scipy.sparse as csr
from random import shuffle
from random import seed
from my_helpers import parse_row
from my_helpers import compute_baselines
from my_helpers import split_data
from my_helpers import calculate_averages
from my_helpers import construct_data
from ALS import run_ALS
from ALS import create_ALS_pred

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


    X_train_userId, X_train_movieId, X_train_rating, X_test_userId, X_test_movieId, X_test_rating = split_data(indices_to_shuffle, userId, movieId, rating, 70)

    full_data = csr.csr_matrix((rating, (userId, movieId)), shape=(10000, 1000)).transpose()
    data_XTrain = csr.csr_matrix((X_train_rating, (X_train_userId, X_train_movieId)), shape=(10000, 1000))\
        .transpose()
    data_XTest = csr.csr_matrix((X_test_rating, (X_test_userId, X_test_movieId)), shape=(10000, 1000))\
        .transpose()

    full_data.eliminate_zeros()
    data_XTrain.eliminate_zeros()
    data_XTest.eliminate_zeros()
    global_avg = data_XTrain[data_XTrain.nonzero()].mean()

    create_ALS_pred(full_data, sub_data)

if __name__ == '__main__':
    main()
